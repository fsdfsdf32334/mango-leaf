let model;
let activeImageUrl;

const classes = [
    {
        name: 'โรคแอนแทรคโนส',
        latin: 'Anthracnose',
        advice: 'มักพบเป็นจุดสีน้ำตาลเข้มหรือดำบนใบ ควรแยกใบที่มีอาการรุนแรงและลดความชื้นสะสมในทรงพุ่ม',
    },
    {
        name: 'โรคแคงเกอร์แบคทีเรีย',
        latin: 'Bacterial Canker',
        advice: 'สังเกตแผลฉ่ำน้ำหรือจุดขอบชัดบนใบ ควรหลีกเลี่ยงการให้น้ำโดนใบและทำความสะอาดอุปกรณ์ตัดแต่ง',
    },
    {
        name: 'แมลงด้วงงวงกัดใบ',
        latin: 'Cutting Weevil',
        advice: 'อาจพบรอยกัดหรือใบถูกทำลายเป็นส่วน ๆ ควรสำรวจใต้ใบและบริเวณยอดอ่อนเพิ่มเติม',
    },
    {
        name: 'โรคกิ่งแห้ง',
        latin: 'Die Back',
        advice: 'มักสัมพันธ์กับยอดหรือกิ่งแห้งลามลงมา ควรตัดแต่งส่วนเสียหายและกำจัดออกจากแปลง',
    },
    {
        name: 'แมลงบั่ว',
        latin: 'Gall Midge',
        advice: 'อาจพบปุ่มนูนหรือรอยผิดปกติบนใบ ควรตรวจใบอ่อนและยอดอ่อนซ้ำในหลายตำแหน่ง',
    },
    {
        name: 'ใบปกติ',
        latin: 'Healthy',
        advice: 'ภาพนี้มีลักษณะใกล้เคียงใบปกติ ควรติดตามอาการเป็นระยะและถ่ายภาพใหม่หากเริ่มพบจุดหรือรอยโรค',
    },
    {
        name: 'โรคราแป้ง',
        latin: 'Powdery Mildew',
        advice: 'มักเห็นคราบคล้ายผงสีขาวบนผิวใบ ควรเพิ่มการถ่ายเทอากาศและลดความชื้นบริเวณทรงพุ่ม',
    },
    {
        name: 'โรคราดำ',
        latin: 'Sooty Mould',
        advice: 'มักเกิดเป็นคราบดำบนใบและสัมพันธ์กับแมลงดูดน้ำเลี้ยง ควรตรวจเพลี้ยหรือแมลงที่สร้างน้ำหวานร่วมด้วย',
    },
];

const imageUpload = document.getElementById('imageUpload');
const imageContainer = document.getElementById('imageContainer');
const predictionClass = document.getElementById('predictionClass');
const predictionAdvice = document.getElementById('predictionAdvice');
const predictionResult = document.getElementById('predictionResult');
const modelStatus = document.getElementById('modelStatus');
const confidenceBadge = document.getElementById('confidenceBadge');
const uploadZone = document.getElementById('uploadZone');

const modelReady = loadModel();

async function loadModel() {
    try {
        setModelStatus('กำลังโหลดโมเดล');
        if (!window.tf) {
            throw new Error('TensorFlow.js is not available');
        }
        model = await tf.loadLayersModel('./model_mango/model.json');
        setModelStatus('พร้อมใช้งาน', 'ready');
        return model;
    } catch (error) {
        console.error('Model failed to load', error);
        setModelStatus('โหลดไม่สำเร็จ', 'error');
        setConfidenceBadge('มีปัญหา', 'error');
        predictionClass.textContent = 'ไม่สามารถโหลดโมเดลได้';
        predictionAdvice.textContent = 'โปรดตรวจสอบการเชื่อมต่ออินเทอร์เน็ตสำหรับ TensorFlow.js และเปิดหน้านี้ผ่าน local web server';
        predictionResult.innerHTML = '<p class="message">ตรวจสอบว่าไฟล์โมเดลอยู่ในโฟลเดอร์ model_mango ครบถ้วน</p>';
        return null;
    }
}

function setModelStatus(text, state = '') {
    modelStatus.textContent = text;
    modelStatus.className = state ? `status-pill ${state}` : 'status-pill';
}

function setConfidenceBadge(text, state = '') {
    confidenceBadge.textContent = text;
    confidenceBadge.className = state ? `confidence-badge ${state}` : 'confidence-badge';
}

function getConfidenceState(probability) {
    if (probability >= 0.8) {
        return { label: 'ความมั่นใจสูง', state: 'high' };
    }

    if (probability >= 0.55) {
        return { label: 'ความมั่นใจปานกลาง', state: 'medium' };
    }

    return { label: 'ควรถ่ายภาพใหม่', state: 'low' };
}

async function predictImage(imageElement) {
    await modelReady;
    if (!model) {
        return;
    }

    predictionClass.textContent = 'กำลังวิเคราะห์ภาพ';
    predictionAdvice.textContent = 'ระบบกำลังอ่านลักษณะใบและคำนวณความเป็นไปได้';
    predictionResult.innerHTML = '<p class="message">กำลังประมวลผล...</p>';
    setConfidenceBadge('กำลังวิเคราะห์');

    const imageTensor = tf.tidy(() => {
        const pixels = tf.browser.fromPixels(imageElement);
        const resized = tf.image.resizeBilinear(pixels, [224, 224]);
        return resized.toFloat().expandDims(0).div(255.0);
    });

    const predictionTensor = model.predict(imageTensor);
    const tensor = Array.isArray(predictionTensor) ? predictionTensor[0] : predictionTensor;
    const predictions = Array.from(await tensor.data());

    imageTensor.dispose();
    tf.dispose(predictionTensor);

    displayPrediction(predictions);
}

function displayPrediction(predictions) {
    const maxProbability = Math.max(...predictions);
    const maxIndex = predictions.indexOf(maxProbability);
    const topClass = classes[maxIndex];
    const confidence = getConfidenceState(maxProbability);

    predictionClass.textContent = `${topClass.name} ${(maxProbability * 100).toFixed(2)}%`;
    predictionAdvice.textContent = topClass.advice;
    setConfidenceBadge(confidence.label, confidence.state);
    predictionResult.innerHTML = '';

    predictions
        .map((probability, index) => ({
            ...classes[index],
            probability,
        }))
        .sort((a, b) => b.probability - a.probability)
        .forEach(({ name, latin, probability }, index) => {
            const percent = (probability * 100).toFixed(2);
            const item = document.createElement('div');
            item.className = index === 0 ? 'probability-item top' : 'probability-item';

            const label = document.createElement('div');
            label.className = 'probability-name';

            const title = document.createElement('strong');
            title.textContent = name;

            const subtitle = document.createElement('span');
            subtitle.textContent = latin;

            label.append(title, subtitle);

            const track = document.createElement('div');
            track.className = 'probability-track';

            const fill = document.createElement('div');
            fill.className = 'probability-fill';
            fill.style.setProperty('--value', `${Math.max(probability * 100, 1)}%`);
            track.appendChild(fill);

            const value = document.createElement('div');
            value.className = 'probability-value';
            value.textContent = `${percent}%`;

            item.append(label, track, value);
            predictionResult.appendChild(item);
        });
}

function renderImagePreview(file) {
    if (!file.type.startsWith('image/')) {
        predictionClass.textContent = 'ไฟล์นี้ไม่ใช่รูปภาพ';
        predictionAdvice.textContent = 'โปรดเลือกไฟล์ภาพนามสกุล JPG, PNG หรือ WebP';
        setConfidenceBadge('เลือกไฟล์ใหม่', 'low');
        return;
    }

    if (activeImageUrl) {
        URL.revokeObjectURL(activeImageUrl);
    }

    activeImageUrl = URL.createObjectURL(file);
    imageContainer.innerHTML = '';

    const imgElement = document.createElement('img');
    imgElement.src = activeImageUrl;
    imgElement.alt = file.name || 'ภาพใบมะม่วงที่อัปโหลด';
    imgElement.onload = () => predictImage(imgElement).catch((error) => {
        console.error('Prediction failed', error);
        predictionClass.textContent = 'วิเคราะห์ภาพไม่สำเร็จ';
        predictionAdvice.textContent = 'โปรดลองรีเฟรชหน้าเว็บหรือเลือกภาพอื่น';
        setConfidenceBadge('มีปัญหา', 'error');
        predictionResult.innerHTML = '<p class="message">ไม่สามารถประมวลผลภาพนี้ได้</p>';
    });

    imageContainer.appendChild(imgElement);
}

imageUpload.addEventListener('change', (event) => {
    const imageFile = event.target.files[0];

    if (!imageFile) {
        return;
    }

    renderImagePreview(imageFile);
});

['dragenter', 'dragover'].forEach((eventName) => {
    uploadZone.addEventListener(eventName, (event) => {
        event.preventDefault();
        uploadZone.classList.add('is-dragging');
    });
});

['dragleave', 'drop'].forEach((eventName) => {
    uploadZone.addEventListener(eventName, (event) => {
        event.preventDefault();
        uploadZone.classList.remove('is-dragging');
    });
});

uploadZone.addEventListener('drop', (event) => {
    const imageFile = event.dataTransfer.files[0];

    if (!imageFile) {
        return;
    }

    imageUpload.files = event.dataTransfer.files;
    renderImagePreview(imageFile);
});
