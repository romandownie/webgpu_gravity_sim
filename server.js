const express = require('express');
const app = express();
const port = 3000; // or any other port you prefer

app.use(express.static('./'));

app.listen(port, () => {
    console.log(`Server is listening at http://localhost:${port}`);
});
