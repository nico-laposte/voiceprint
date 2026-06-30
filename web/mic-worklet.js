/**
 * AudioWorkletProcessor qui tourne sur le thread audio dédié (pas le
 * thread principal -> pas de freeze UI). Il accumule les samples bruts
 * du micro et les envoie au thread principal par blocs de taille fixe,
 * qui se chargera de les transmettre au serveur via WebSocket.
 */
class MicCaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    // ~250ms de buffer avant envoi (à 16kHz mono) : compromis entre
    // fréquence des messages WebSocket et latence d'accumulation.
    this.chunkSize = 4000;
    this.buffer = new Float32Array(this.chunkSize);
    this.writeIndex = 0;
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || input.length === 0) return true;
    const channelData = input[0]; // mono : on ne prend que le 1er canal

    for (let i = 0; i < channelData.length; i++) {
      this.buffer[this.writeIndex++] = channelData[i];
      if (this.writeIndex === this.chunkSize) {
        // On envoie une copie (le buffer sera réutilisé tout de suite après)
        this.port.postMessage(this.buffer.slice(0));
        this.writeIndex = 0;
      }
    }
    return true;
  }
}

registerProcessor('mic-capture-processor', MicCaptureProcessor);
