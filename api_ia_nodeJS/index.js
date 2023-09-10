const express = require('express');
const canvas = require('canvas');
const faceapi = require('face-api.js');
const multer = require('multer');
const { Canvas, Image, ImageData } = canvas;


const app = express();
const port = process.env.PORT || 3001;

app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ limit: '50mb', extended: true }));

// Configuración de multer para manejar la carga de archivos
const storage = multer.memoryStorage();
const upload = multer({ storage: storage });

// Configuración de face-api.js
faceapi.env.monkeyPatch({ Canvas, Image, ImageData });
const MODEL_URL = './models';
const faceExpressionNet = faceapi.nets.faceExpressionNet;


// Este es el original pero modificamos las expresiones a las que vamos a usar
// const emotionTranslations = {
//   neutral: 'Neutral',
//   happy: 'Feliz',
//   sad: 'Triste',
//   angry: 'Enojado',
//   fearful: 'Asustado',
//   disgusted: 'Disgustado',
//   surprised: 'Sorprendido'
// };

const emotionTranslations = {
  neutral: 'Calma',
  happy: 'Alegria',
  sad: 'Tristeza',
  angry: 'Enojo',
  fearful: 'Tristeza',
  disgusted: 'Tristeza',
  surprised: 'Sorpresa'
};

Promise.all([
  faceExpressionNet.loadFromDisk(MODEL_URL),
  faceapi.nets.ssdMobilenetv1.loadFromDisk(MODEL_URL)
])
  .then(() => {
    console.log('Modelos cargados. El servidor está listo.');
    startServer();
  })
  .catch((error) => {
    console.error('Error al cargar los modelos:', error);
  });

function startServer() {
  app.listen(port, () => {
    console.log(`El servidor está escuchando en el puerto ${port}`);
  });
}

const maxImageCount = 20;
app.use('/detect-emotion', upload.array('images', maxImageCount));
app.post('/detect-emotion', async (req, res) => {
  const images = req.files;

  if (!images || images.length === 0) {
    return res.status(400).json({ error: 'No se han enviado imágenes' });
  }

  try {
    const emotionsResponses = {};

    for (let i = 0; i < images.length; i++) {
      const image = images[i];
      const imageBuffer = image.buffer;
      const imageResult = {};

      const img = new Image();
      img.src = imageBuffer;

      // Detección de emociones
      const detections = await faceapi.detectSingleFace(img).withFaceExpressions();

      if (!detections) {
        imageResult.emotion = 'No se detectó ninguna cara en la imagen.';
      } else {
        const emotions = detections.expressions;
        const highestEmotion = Object.keys(emotions).reduce((a, b) => (emotions[a] > emotions[b] ? a : b));
        imageResult.emotion = emotionTranslations[highestEmotion] || highestEmotion;
      }

      emotionsResponses[`emotion_${i + 1}`] = imageResult.emotion;
    }

    res.json(emotionsResponses);
  } catch (error) {
    console.error('Error al cargar o procesar las imágenes:', error);
    return res.status(500).json({ error: 'Error al procesar las imágenes.' });
  }
});