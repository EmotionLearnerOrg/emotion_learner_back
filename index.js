const express = require('express');
const canvas = require('canvas');
const faceapi = require('face-api.js');

const { Canvas, Image, ImageData } = canvas;
const jimp = require('jimp');

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
])
  .then(() => {
    console.log('Modelos cargados. El servidor está listo.');
    // Iniciar el servidor después de cargar los modelos
    startServer();
  })
  .catch((error) => {
    console.error('Error al cargar los modelos:', error);
  });

function startServer() {

  app.listen(port, () => {
    console.log(`El servidor está escuchando en el puerto ${port}`);
  });

  app.post('/detect-emotion', async (req, res) => {
    const base64Data = req.body.image.replace(/^data:image\/png;base64,/, '');
    const image = new Image();

    try {
      const jimpImage = await jimp.read(Buffer.from(base64Data, 'base64'));
      image.src = await jimpImage.getBase64Async(jimp.MIME_PNG); // Convierto el objeto Jimp a HTMLImageElement para faceapi
      console.log('Imagen cargada con éxito');
    } catch (error) {
      console.error('Error al cargar la imagen:', error);
    }

    // Detección de emociones
    const detections = await faceapi.detectAllFaces(image).withFaceExpressions();

    if (detections.length === 0) {
      res.status(400).json({ error: 'No se detectó ninguna cara en la imagen.' });
    } else {
      const emotions = detections[0].expressions;
      res.json({ emotions });
    }
    
  });
}