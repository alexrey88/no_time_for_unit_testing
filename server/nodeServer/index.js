var fs = require('fs');
var multer = require('multer');
var upload = multer({ dest: 'uploads/' }); //setting the default folder for multer

const express = require('express')
const app = express()
const port = 3000

app.get('/', (req, res) => res.send('Hello World!'))

app.get('/test', (req, res) => res.send('test!'))

app.get('/lat/:lat/lng/:lng', function (req, res) {
    console.log(req.params)
    console.log('Coordinates called')
    res.send(req.params)
})

app.get('/img/:img', function (req, res) {
    console.log('Img called')
    res.send(req.params)
})

app.post('/upload', upload.single('fileData'), (req, res, next) => {
    res.send('Thanks')
    console.log('Upload called')
    fs.readFile(req.file.path, (err, contents) => {
        if (err) {
            console.log('Error: ', err);
        } else {
            console.log('File contents ', contents);
        }
    });
});

app.listen(port, () => console.log(`Server listening on port ${port}...`))

