var fs = require('fs');
var multer = require('multer');
var upload = multer({ dest: 'uploads/' });

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
    
    console.log('Upload called')
    fs.readFile(req.file.path, (err, contents) => {
        if (err) {
            console.log('Error: ', err);
        } else {
            console.log('File contents ', contents);
        }
    });

    /* Calling python script */
    const spawn = require('child_process').spawn;
    const ls = spawn('python', ['script.py', 'arg1', 'arg2']);
    ls.stdout.on('data', (data) => {
        console.log(`stdout: ${data}`);
      });
      
      ls.stderr.on('data', (data) => {
        console.log(`stderr: ${data}`);
      });
      
      ls.on('close', (code) => {
        console.log(`child process exited with code ${code}`);
      });
    res.send('Ran script')
    /* End of python */

    res.send('Thanks')
});

app.get('/getData', function (req, res) {
    const spawn = require('child_process').spawn;
    const ls = spawn('python', ['script.py', 'arg1', 'arg2']);
    ls.stdout.on('data', (data) => {
        console.log(`stdout: ${data}`);
      });
      
      ls.stderr.on('data', (data) => {
        console.log(`stderr: ${data}`);
      });
      
      ls.on('close', (code) => {
        console.log(`child process exited with code ${code}`);
      });
    res.send('Ran script')
})

app.listen(port, () => console.log(`Server listening on port ${port}...`))

