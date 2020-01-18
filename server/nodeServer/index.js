const express = require('express')
const app = express()
const port = 3000

app.get('/', (req, res) => res.send('Hello World!'))

app.get('/test', (req, res) => res.send('test!'))

app.get('/lat/:lat/lng/:lng', function (req, res) {
    console.log(req.params)
    res.send(req.params)
})

app.listen(port, () => console.log(`Server listening on port ${port}...`))

