const express = require('express');
const fs = require('fs');
const canvas = require('canvas');
const faceapi = require('face-api.js');
const { Canvas, Image, ImageData } = canvas;

const app = express();
const port = process.env.PORT || 3001;

app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ limit: '50mb', extended: true }));

// Configuración de face-api.js
faceapi.env.monkeyPatch({ Canvas, Image, ImageData });
const MODEL_URL = './models';
const emotionModel = faceapi.nets.ssdMobilenetv1;
const faceExpressionNet = faceapi.nets.faceExpressionNet;

Promise.all([
  emotionModel.loadFromDisk(MODEL_URL),
  faceExpressionNet.loadFromDisk(MODEL_URL),
]).then(startServer);

function startServer() {
  console.log('Modelos cargados. El servidor está listo.');

  // Ruta de la API
  app.post('/detect-emotion', async (req, res) => {
    const base64Data = req.body.image.replace(/^data:image\/png;base64,/, '');

    fs.writeFileSync('image.png', base64Data, 'base64', function (err) {
      if (err) {
        console.error(err);
        res.status(500).json({ error: 'Error al guardar la imagen.' });
      }
    });

    const img = await canvas.loadImage('./image.png');

    // Detección de emociones
    const detections = await faceapi
      .detectAllFaces(img)
      .withFaceExpressions();

    if (detections.length === 0) {
      res.status(400).json({ error: 'No se detectó ninguna cara en la imagen.' });
    } else {
      const emotions = detections[0].expressions;
      res.json({ emotions });
    }
  });

  app.listen(port, () => {
    console.log(`El servidor está escuchando en el puerto ${port}`);
  });
}
