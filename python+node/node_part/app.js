const tf = require('@tensorflow/tfjs-node');
const bodyPix = require('@tensorflow-models/body-pix');
const http = require('http');

const net = bodyPix.load({
    architecture: 'MobileNetV1',
    outputStride: 16,
    multiplier: 0.75,
    quantBytes: 2
});
http.createServer(function(req, res){
    res.writeHead(200, { "Content-Type": "text/plain" });

    res.end("Hello World!\n");
}).listen(8080, "127.0.0.1");