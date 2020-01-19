var fs = require('fs');
var multer = require('multer');
var upload = multer({ dest: 'uploads/' });

const express = require('express')
const app = express()
const port = 3000

app.get('/getDatabase', function (req, res) {
    console.log('GET DATABASE')
    let rawdb = fs.readFileSync('db.json');
    let db = JSON.parse(rawdb);
    console.log(db)
    res.send(db)
})

app.get('/', (req, res) => res.send('Hello World!'))

app.get('/test', (req, res) => res.send('test!'))

app.get('/lat/:lat/lng/:lng', function (req, res) {
    console.log(req.params)
    console.log('Coordinates called')



    /* Write coordinates */
    let entry = {
        lat: req.params.lat,
        lng: req.params.lng
    };
    console.log(entry)

    let dataToWrite = JSON.stringify(entry);
    fs.writeFileSync('coordinates.json', dataToWrite);

    ///////

    res.send(req.params)
})

// app.get('/img/:img', function (req, res) {
//     console.log('Img called')
//     res.send(req.params)
// })

// app.post('newEntry/lat/:lat/lng/:lng', upload.single('fileData'), (req, res, next) => {

//     console.log('Upload 2.0 called')
//     fs.readFile(req.file.path, (err, contents) => {
//         if (err) {
//             console.log('Error: ', err);
//         } else {
//             console.log('File contents ', contents);
//         }
//     });

//     /* Calling python script */
//     const spawn = require('child_process').spawn;
//     const ls = spawn('python', ['script.py', 'arg1', 'arg2']);

//     // other, trash, bench, picnic
//     var type = 'other';

//     ls.stdout.on('data', (data) => {
//         console.log(`stdout: ${data}`);
//     });

//     ls.stderr.on('data', (data) => {
//         console.log(`stderr: ${data}`);
//         type = `${data}`
//         console.log('type:')
//         console.log(type)

//         /* Writing to DB */

//         let entry = { 
//             type: type,
//             lat: req.params.lat,
//             lng: req.params.lng
//         };
//         console.log(entry)

//         // let data = JSON.stringify(student);
//         // fs.writeFileSync('student-2.json', data);

//         /******/
//         res.send({ 'data': type })
//     });

//     ls.on('close', (code) => {
//         console.log(`child process exited with code ${code}`);
//     });
//     // res.send('Ran script')
//     /* End of python */


// });

// app.get('/getData', function (req, res) {
//     const spawn = require('child_process').spawn;
//     const ls = spawn('python', ['script.py', 'arg1', 'arg2']);

//     // other, trash, bench, picnic
//     var type = 'other';

//     ls.stdout.on('data', (data) => {
//         console.log(`stdout: ${data}`);
//     });

//     ls.stderr.on('data', (data) => {
//         console.log(`stderr: ${data}`);
//         type = `${data}`
//         console.log('type:')
//         console.log(type)
//         res.send(type)
//     });

//     ls.on('close', (code) => {
//         console.log(`child process exited with code ${code}`);
//     });
//     // res.send('Ran script')
//     /* End of python */


// })

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

    // other, trash, bench, picnic
    var type = 'other';

    ls.stdout.on('data', (data) => {
        console.log(`stdout: ${data}`);
    });

    ls.stderr.on('data', (data) => {
        console.log(`stderr: ${data}`);
        type = `${data}`
        console.log('type:')
        console.log(type)

        /* Read coordinates */
        let rawdata = fs.readFileSync('coordinates.json');
        let coordinates = JSON.parse(rawdata);
        console.log(coordinates);

        lat = coordinates.lat
        lng = coordinates.lng

        /* Read DB */
        let rawdb = fs.readFileSync('db.json');
        let db = JSON.parse(rawdb);
        console.log(db)

        /* Update DB */
        var myList = db.list
        let entry = {
            type: type,
            lat: lat,
            lng: lng
        };
        console.log(entry)
        myList.push(entry)
        newDb = { list: myList }

        /* Writing to DB */
        let dataToWrite = JSON.stringify(newDb);
        fs.writeFileSync('db.json', dataToWrite);

        /******/

        res.send({ 'data': type })
    });

    ls.on('close', (code) => {
        console.log(`child process exited with code ${code}`);
    });
    // res.send('Ran script')
    /* End of python */


});

app.listen(port, () => console.log(`Server listening on port ${port}...`))

