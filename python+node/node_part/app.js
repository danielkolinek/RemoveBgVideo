const tf = require('@tensorflow/tfjs-node-gpu');
const bodyPix = require('@tensorflow-models/body-pix');
const http = require('http');
(async () => {
    //mobilnet
    /* 
    const net = await bodyPix.load({
        architecture: 'MobileNetV1',
        outputStride: 16,
        multiplier: 0.75,
        quantBytes: 2
    });
    */
   //resnet
   /*
    const net = await bodyPix.load({
        architecture: "ResNet50",
        quantBytes: 1,
        outputStride: 16
    });
    */
    const net = await bodyPix.load({
        architecture: 'MobileNetV1',
        outputStride: 16,
        multiplier: 0.75,
        quantBytes: 2
    });
    http.createServer(function(req, res){
        var images = [];
        req.on('data', (image) => {
            images.push(image);
        });
        req.on('end', async () => {
            const image = tf.node.decodeImage(Buffer.concat(images));
            segmentation = await net.segmentPerson(image, {
                flipHorizontal: false,
                internalResolution: 'medium',
                segmentationThreshold: 0.75,
            });
            res.writeHead(200, { 'Content-Type': 'application/octet-stream' });
            res.write(Buffer.from(segmentation.data));
            res.end();
            tf.dispose(image);
        });
    }).listen(8080);;
})();