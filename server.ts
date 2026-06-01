import "dotenv/config";
import express from "express";
import path from "path";
import { createServer as createViteServer } from "vite";
import { GoogleGenAI, Modality } from "@google/genai";

let lamejs: any;

async function startServer() {
  (global as any).window = global;
  lamejs = await import("lamejs").then(m => m.default || m);

  const app = express();
  const PORT = process.env.PORT || 3000;

  app.use(express.json());

  let aiClient: GoogleGenAI | null = null;
  function getAIClient() {
    if (!aiClient) {
      const key = process.env.GEMINI_API_KEY;
      if (!key || key.trim() === "" || key === "MY_GEMINI_API_KEY") {
        throw new Error('GEMINI_API_KEY environment variable is missing or set to a placeholder. Please configure your Gemini API Key in Settings > Secrets.');
      }
      aiClient = new GoogleGenAI({ 
        apiKey: key,
        httpOptions: {
          headers: {
            'User-Agent': 'aistudio-build',
          }
        }
      });
    }
    return aiClient;
  }

  function pcmToMp3Blob(pcmData: Uint8Array, sampleRate: number): Blob {
    let Mp3Encoder = (lamejs as any).Mp3Encoder;
    if (!Mp3Encoder && (lamejs as any).default) {
      Mp3Encoder = (lamejs as any).default.Mp3Encoder;
    }
    if (!Mp3Encoder) {
      throw new Error("Mp3Encoder not found in lamejs library");
    }
    const mp3encoder = new Mp3Encoder(1, sampleRate, 128);
    let buffer = pcmData.buffer;
    let offset = pcmData.byteOffset;
    if (offset % 2 !== 0) {
      const aligned = new Uint8Array(pcmData.length);
      aligned.set(pcmData);
      buffer = aligned.buffer;
      offset = 0;
    }
    const samples = new Int16Array(buffer, offset, Math.floor(pcmData.byteLength / 2));
    const mp3Data: Uint8Array[] = [];
    const sampleBlockSize = 1152;
    for (let i = 0; i < samples.length; i += sampleBlockSize) {
      const chunk = samples.subarray(i, i + sampleBlockSize);
      const mp3buf = mp3encoder.encodeBuffer(chunk);
      if (mp3buf.length > 0) {
        mp3Data.push(new Uint8Array(mp3buf));
      }
    }
    const endBuffer = mp3encoder.flush();
    if (endBuffer.length > 0) {
      mp3Data.push(new Uint8Array(endBuffer));
    }
    return new Blob(mp3Data, { type: 'audio/mpeg' });
  }

  async function produceDialogue(segments: any[], dailyUsage: number, userLimit: number): Promise<{ wav: Blob; mp3?: Blob }> {
    const validSegments = segments.filter(s => s.text && s.text.trim());
    if (validSegments.length === 0) {
      throw new Error("No valid segments with text provided");
    }

    // Call GenAI TTS for all segments in parallel to avoid sequential timing timeouts
    const promises = validSegments.map(async (segment) => {
      const response = await getAIClient().models.generateContent({
        model: "gemini-3.1-flash-tts-preview",
        contents: [{ parts: [{ text: segment.text }] }],
        config: {
          responseModalities: [Modality.AUDIO],
          speechConfig: {
            voiceConfig: {
              prebuiltVoiceConfig: { voiceName: segment.voiceId },
            },
          },
        },
      });
      const base64Data = response.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
      if (!base64Data) {
        throw new Error(`Failed to produce audio for segment: "${segment.text.slice(0, 20)}..."`);
      }
      return { base64Data };
    });

    const results = await Promise.all(promises);

    const audioChunks: Uint8Array[] = [];
    let currentUsage = dailyUsage;

    for (const res of results) {
      const buffer = Buffer.from(res.base64Data, 'base64');
      const chunkDuration = Math.ceil(buffer.length / 48000);
      if (currentUsage + chunkDuration > userLimit) {
        throw new Error("Your file is more than your limited time. Please upgrade your plan");
      }
      currentUsage += chunkDuration;
      audioChunks.push(new Uint8Array(buffer));
    }

    const totalLen = audioChunks.reduce((acc, chunk) => acc + chunk.length, 0);
    const combinedPCM = new Uint8Array(totalLen);
    let offset = 0;
    for (const chunk of audioChunks) {
      combinedPCM.set(chunk, offset);
      offset += chunk.length;
    }
    const wavBlob = wrapInWavBlob(combinedPCM, 24000);
    let mp3Blob: Blob | undefined;
    try {
      mp3Blob = pcmToMp3Blob(combinedPCM, 24000);
    } catch (e) {
      console.error("Server-side MP3 conversion failed:", e);
    }
    return { wav: wavBlob, mp3: mp3Blob };
  }

  function wrapInWavBlob(pcmData: Uint8Array, sampleRate: number): Blob {
    const len = pcmData.length;
    const header = new ArrayBuffer(44);
    const view = new DataView(header);
    view.setUint32(0, 0x52494646, false); // "RIFF"
    view.setUint32(4, 36 + len, true);
    view.setUint32(8, 0x57415645, false); // "WAVE"
    view.setUint32(12, 0x666d7420, false); // "fmt "
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true); // PCM
    view.setUint16(22, 1, true); // Mono
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    view.setUint32(36, 0x64617461, false); // "data"
    view.setUint32(40, len, true);
    return new Blob([header, pcmData], { type: 'audio/wav' });
  }

  // API routes
  app.post("/api/produce-dialogue", async (req, res) => {
    try {
      const { segments, dailyUsage, userLimit, isAdmin } = req.body;
      const effectiveLimit = isAdmin ? Infinity : (userLimit || 10);
      const { wav, mp3 } = await produceDialogue(segments, dailyUsage, effectiveLimit);
      
      const wavBase64 = Buffer.from(await wav.arrayBuffer()).toString('base64');
      const mp3Base64 = mp3 ? Buffer.from(await mp3.arrayBuffer()).toString('base64') : '';
      
      res.json({ wav: `data:audio/wav;base64,${wavBase64}`, mp3: mp3 ? `data:audio/mpeg;base64,${mp3Base64}` : '' });
    } catch (error: any) {
      console.error(error);
      if (error.message === "Your file is more than your limited time. Please upgrade your plan") {
        res.status(403).json({ error: error.message });
      } else {
        res.status(500).json({ error: error.message || "Failed to produce dialogue" });
      }
    }
  });

  app.post("/api/generate-script", async (req, res) => {
    try {
      const { prompt } = req.body;
      if (!prompt || !prompt.trim()) {
        return res.status(400).json({ error: "Prompt is required" });
      }
      const response = await getAIClient().models.generateContent({
        model: "gemini-3.5-flash",
        contents: `Write a professional voiceover script for the following topic: ${prompt}. Keep it concise and impactful.`,
      });
      res.json({ text: response.text || "Failed to generate script." });
    } catch (error: any) {
      console.error(error);
      res.status(500).json({ error: error.message || "Failed to generate script" });
    }
  });

  // Vite middleware for development
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), 'dist');
    app.use(express.static(distPath));
    app.get('*', (req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server running on http://localhost:${PORT}`);
  });
}

startServer();
