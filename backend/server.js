// express starter

const express = require("express");

const app = express();
const port = 3000;

app.use(express.json());

app.get("/", (req, res) => {
  res.send("Welcome to video translator!");
});

app.use("/api/translate", require("./routes/translate"));

app.listen(port, () => {
  console.log(`Server is running on port ${port}`);
});
