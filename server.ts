import "dotenv/config";
import express from "express";
import { GoogleGenAI, Modality } from "@google/genai";
import lamejs from "lamejs";

async function startServer() {
  const app = express();
  const PORT = process.env.PORT || 3000;

  app.use(express.json());

  let aiClient: GoogleGenAI | null = null;

  function getAIClient() {
    if (!aiClient) {
      const key = process.env.GEMINI_API_KEY;
      if (!key || key.trim() === "" || key === "MY_GEMINI_API_KEY") {
        throw new Error("GEMINI_API_KEY is missing. Set it in Render → Environment.");
      }
      aiClient = new GoogleGenAI({
        apiKey: key,
        httpOptions: {
          headers: { "User-Agent": "aistudio-build" }
        }
      });
    }
    return aiClient;
  }

  function pcmToMp3Buffer(pcmData: Uint8Array, sampleRate: number): Buffer {
    const Mp3Encoder = lamejs.Mp3Encoder;
    const encoder = new Mp3Encoder(1, sampleRate, 128);

    const samples = new Int16Array(pcmData.buffer);
    const mp3Data: Uint8Array[] = [];

    const block = 1152;
    for (let i = 0; i < samples.length; i += block) {
      const chunk = samples.subarray(i, i + block);
      const mp3buf = encoder.encodeBuffer(chunk);
      if (mp3buf.length > 0) mp3Data.push(new Uint8Array(mp3buf));
    }

    const end = encoder.flush();
    if (end.length > 0) mp3Data.push(new Uint8Array(end));

    return Buffer.concat(mp3Data.map(u => Buffer.from(u)));
  }

  function wrapInWavBuffer(pcmData: Uint8Array, sampleRate: number): Buffer {
    const header = Buffer.alloc(44);
    const dataSize = pcmData.length;

    header.write("RIFF", 0);
    header.writeUInt32LE(36 + dataSize, 4);
    header.write("WAVE", 8);
    header.write("fmt ", 12);
    header.writeUInt32LE(16, 16);
    header.writeUInt16LE(1, 20);
    header.writeUInt16LE(1, 22);
    header.writeUInt32LE(sampleRate, 24);
    header.writeUInt32LE(sampleRate * 2, 28);
    header.writeUInt16LE(2, 32);
    header.writeUInt16LE(16, 34);
    header.write("data", 36);
    header.writeUInt32LE(dataSize, 40);

    return Buffer.concat([header, Buffer.from(pcmData)]);
  }

  async function produceDialogue(segments: any[], dailyUsage: number, userLimit: number) {
    const valid = segments.filter(s => s.text?.trim());
    if (valid.length === 0) throw new Error("No valid segments");

    const results = await Promise.all(
      valid.map(async seg => {
        const res = await getAIClient().models.generateContent({
          model: "gemini-3.1-flash-tts-preview",
          contents: [{ parts: [{ text: seg.text }] }],
          config: {
            responseModalities: [Modality.AUDIO],
            speechConfig: {
              voiceConfig: { prebuiltVoiceConfig: { voiceName: seg.voiceId } }
            }
          }
        });

        const base64 = res.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
        if (!base64) throw new Error("Audio generation failed");

        return Buffer.from(base64, "base64");
      })
    );

    const totalPCM = Buffer.concat(results);
    const wav = wrapInWavBuffer(new Uint8Array(totalPCM), 24000);
    const mp3 = pcmToMp3Buffer(new Uint8Array(totalPCM), 24000);

    return { wav, mp3 };
  }

  app.post("/api/produce-dialogue", async (req, res) => {
    try {
      const { segments, dailyUsage, userLimit, isAdmin } = req.body;
      const limit = isAdmin ? Infinity : userLimit || 10;

      const { wav, mp3 } = await produceDialogue(segments, dailyUsage, limit);

      res.json({
        wav: `data:audio/wav;base64,${wav.toString("base64")}`,
        mp3: `data:audio/mpeg;base64,${mp3.toString("base64")}`
      });
    } catch (err: any) {
      res.status(500).json({ error: err.message });
    }
  });

  app.post("/api/generate-script", async (req, res) => {
    try {
      const { prompt } = req.body;
      if (!prompt?.trim()) return res.status(400).json({ error: "Prompt required" });

      const response = await getAIClient().models.generateContent({
        model: "gemini-3.5-flash",
        contents: `Write a professional voiceover script: ${prompt}`
      });

      res.json({ text: response.text || "" });
    } catch (err: any) {
      res.status(500).json({ error: err.message });
    }
  });

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server running on port ${PORT}`);
  });
}

startServer();

