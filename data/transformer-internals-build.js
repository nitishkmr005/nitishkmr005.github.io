const path = require('path');
const pptxgen = require('pptxgenjs');
const html2pptx = require(path.join(__dirname, 'html2pptx-local'));

const slidesDir = path.join(__dirname, 'transformer-internals-slides');
const outputFile = path.join(__dirname, 'transformer-internals-what-changed-since-2017.pptx');

const slideFiles = [
  'slide-01.html',
  'slide-02.html',
  'slide-03.html',
  'slide-13.html',
  'slide-04.html',
  'slide-05.html',
  'slide-19.html',
  'slide-06.html',
  'slide-14.html',
  'slide-07.html',
  'slide-15.html',
  'slide-25.html',
  'slide-08.html',
  'slide-16.html',
  'slide-24.html',
  'slide-09.html',
  'slide-17.html',
  'slide-18.html',
  'slide-10.html',
  'slide-20.html',
  'slide-11.html',
  'slide-21.html',
  'slide-12.html',
  'slide-22.html',
  'slide-23.html'
];

async function build() {
  const pptx = new pptxgen();
  pptx.layout = 'LAYOUT_16x9';
  pptx.author = 'Transformer Internals';
  pptx.title = 'Transformer Internals: What Actually Changed Since 2017';

  for (const file of slideFiles) {
    const htmlPath = path.join(slidesDir, file);
    await html2pptx(htmlPath, pptx);
  }

  await pptx.writeFile({ fileName: outputFile });
}

build().catch((err) => {
  console.error(err);
  process.exit(1);
});
