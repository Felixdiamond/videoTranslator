const express = require('express');
const router = express.Router();

const { entryFunction } = require('../controllers/translationController');

router.post('/', entryFunction);

module.exports = router;