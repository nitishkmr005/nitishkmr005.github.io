const pptxgen = require('pptxgenjs');
const html2pptx = require('/Users/nitishkumarharsoor/Documents/1.Learnings/1.Projects/4.Experiments/1.portfolio/.claude/skills/pptx/scripts/html2pptx.js');
const path = require('path');

async function createPresentation() {
    const pptx = new pptxgen();
    pptx.layout = 'LAYOUT_16x9';
    pptx.author = 'Professional Presentation';
    pptx.title = 'Transformer Internals: What Changed Since 2017';
    pptx.subject = 'Technical Deep Dive';

    const slidesDir = '/Users/nitishkumarharsoor/Documents/1.Learnings/1.Projects/4.Experiments/1.portfolio/static/files/workspace/slides';

    const slideFiles = [
        'slide1.html',
        'slide2.html',
        'slide3.html',
        'slide4.html',
        'slide5.html',
        'slide6.html',
        'slide7.html',
        'slide8.html',
        'slide9.html'
    ];

    for (const slideFile of slideFiles) {
        const htmlPath = path.join(slidesDir, slideFile);
        console.log(`Processing: ${slideFile}`);
        try {
            await html2pptx(htmlPath, pptx);
            console.log(`  ✓ ${slideFile} added successfully`);
        } catch (err) {
            console.error(`  ✗ Error with ${slideFile}:`, err.message);
        }
    }

    const outputPath = '/Users/nitishkumarharsoor/Documents/1.Learnings/1.Projects/4.Experiments/1.portfolio/static/files/transformer-internals-professional.pptx';
    await pptx.writeFile({ fileName: outputPath });
    console.log(`\nPresentation saved to: ${outputPath}`);
}

createPresentation().catch(console.error);
