const express = require('express');
const cors = require('cors');

const app = express();
app.use(cors()); // Allow frontend to connect
app.use(express.json()); // Parse JSON body

app.post('/suggest', (req, res) => {
  const { soilType, region } = req.body;

  let suggestedCrop = 'rice';
  let reason = 'Default suggestion for general conditions.';

  if (soilType.toLowerCase() === 'sandy' && region.toLowerCase() === 'tamil nadu') {
    suggestedCrop = 'peanuts';
    reason = 'Sandy soil in Tamil Nadu is ideal for peanut cultivation.';
  }

  res.json({
    success: true,
    soilType,
    region,
    suggestedCrop,
    reason
  });
});

app.listen(5000, () => {
  console.log('✅ API is running at http://127.0.0.1:5000');
});