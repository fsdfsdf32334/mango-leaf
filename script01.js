const MODEL_PATH = './model_mango/model.json';
const INPUT_SIZE = [224, 224];
const TFJS_SOURCES = [
    'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.19.0/dist/tf.min.js',
    'https://unpkg.com/@tensorflow/tfjs@4.19.0/dist/tf.min.js',
];
const LOW_CONFIDENCE_THRESHOLD = 0.55;
const HIGH_CONFIDENCE_THRESHOLD = 0.8;
const HISTORY_KEY = 'mangocare-scan-history-v1';
const IS_DEVELOPMENT = location.hostname === 'localhost' || location.hostname === '127.0.0.1';
const DEFAULT_RESULT_EMPTY_HTML = `
    <p class="eyebrow">Result</p>
    <h2 id="resultTitle">ผลการประเมินจะแสดงที่นี่</h2>
    <p>ระบบนี้เป็นเครื่องมือช่วยคัดกรองจากภาพถ่าย ไม่ใช่การวินิจฉัยยืนยัน ควรใช้ร่วมกับการสำรวจแปลงและคำแนะนำจากผู้เชี่ยวชาญ</p>
`;

let model = null;
let modelPromise = null;
let currentImageUrl = '';
let currentImageFile = null;
let isPredicting = false;
let showAllPredictions = false;
let latestPredictions = [];
let pendingConfirmResolve = null;

// Class order follows train_generator.class_indices in MobileNet_Final.ipynb.
const classes = [
    {
        id: 'anthracnose',
        thaiName: 'โรคแอนแทรคโนส',
        englishName: 'Anthracnose',
        category: 'Disease',
        shortDescription: 'ลักษณะใกล้เคียงโรคที่มักพบเป็นจุดหรือแผลสีน้ำตาลเข้มถึงดำบนใบ โดยควรตรวจร่วมกับสภาพแปลงจริง',
        symptoms: ['จุดสีน้ำตาลหรือดำบนผิวใบ', 'แผลอาจขยายและมีขอบค่อนข้างชัด', 'ใบที่มีอาการมากอาจแห้งหรือเสียรูป'],
        monitoringTips: ['สำรวจใบหลายตำแหน่งในทรงพุ่ม', 'ตรวจว่าพบอาการคล้ายกันบนใบอ่อนหรือใบแก่หรือไม่', 'บันทึกภาพซ้ำเมื่อสภาพแสงใกล้เคียงกันเพื่อเทียบการเปลี่ยนแปลง'],
        preventionTips: ['ลดความชื้นสะสมด้วยการจัดทรงพุ่มให้โปร่ง', 'เก็บใบหรือเศษพืชที่เสียหายออกจากแปลงอย่างเหมาะสม', 'ปรึกษาเจ้าหน้าที่เกษตรเมื่ออาการลุกลามหรือพบหลายต้น'],
    },
    {
        id: 'bacterial-canker',
        thaiName: 'โรคแคงเกอร์แบคทีเรีย',
        englishName: 'Bacterial Canker',
        category: 'Disease',
        shortDescription: 'ผลประเมินชี้ว่าภาพมีลักษณะใกล้เคียงแผลหรือจุดผิดปกติที่อาจเกี่ยวข้องกับอาการแคงเกอร์ ควรตรวจแปลงเพิ่มเติม',
        symptoms: ['พบจุดหรือแผลขอบชัดบนใบ', 'บางกรณีแผลดูฉ่ำน้ำหรือมีวงรอบแผล', 'อาการอาจพบร่วมกับกิ่งหรือส่วนอื่นของต้น'],
        monitoringTips: ['ตรวจใบใกล้บริเวณที่มีอาการและใบจากต้นข้างเคียง', 'สังเกตว่าจุดแผลเพิ่มจำนวนหลังฝนหรือความชื้นสูงหรือไม่', 'หลีกเลี่ยงการสรุปจากภาพเดียวหากความมั่นใจต่ำ'],
        preventionTips: ['ทำความสะอาดอุปกรณ์ตัดแต่งก่อนใช้กับต้นอื่น', 'หลีกเลี่ยงการให้น้ำกระแทกใบโดยตรงหากทำได้', 'ขอคำแนะนำจากผู้เชี่ยวชาญก่อนเลือกวิธีจัดการเฉพาะทาง'],
    },
    {
        id: 'cutting-weevil',
        thaiName: 'อาการจากด้วงงวงกัดใบ',
        englishName: 'Cutting Weevil',
        category: 'Pest',
        shortDescription: 'ภาพมีลักษณะใกล้เคียงความเสียหายจากแมลงกัดกินใบ ควรสำรวจใบอ่อน ใต้ใบ และยอดอ่อนเพิ่มเติม',
        symptoms: ['ขอบใบหรือแผ่นใบมีรอยถูกกัด', 'ใบอ่อนอาจเสียรูปหรือขาดเป็นส่วน', 'อาจพบร่องรอยแมลงบริเวณยอดหรือใต้ใบ'],
        monitoringTips: ['สำรวจช่วงเช้าหรือเย็นเมื่อแมลงอาจพบได้ง่ายขึ้น', 'ตรวจใบอ่อนและยอดอ่อนหลายตำแหน่ง', 'ดูว่ารอยกัดเป็นรอยใหม่หรือแห้งเก่าแล้ว'],
        preventionTips: ['ลดแหล่งหลบซ่อนด้วยการดูแลแปลงให้สะอาด', 'ติดตามความเสียหายซ้ำก่อนตัดสินใจจัดการ', 'ปรึกษาเจ้าหน้าที่เกษตรเมื่อพบการระบาดต่อเนื่อง'],
    },
    {
        id: 'die-back',
        thaiName: 'โรคกิ่งแห้ง',
        englishName: 'Die Back',
        category: 'Disease',
        shortDescription: 'ลักษณะภาพใกล้เคียงอาการเสื่อมสภาพหรือแห้งของเนื้อเยื่อ ควรตรวจยอด กิ่ง และบริเวณรอยต่อของกิ่งร่วมด้วย',
        symptoms: ['ใบหรือส่วนยอดดูแห้งผิดปกติ', 'อาการอาจลามจากปลายยอดหรือกิ่ง', 'สีใบอาจเปลี่ยนและแห้งกรอบในบางจุด'],
        monitoringTips: ['ตรวจว่ากิ่งหรือยอดใกล้ใบมีอาการแห้งร่วมกันหรือไม่', 'ดูความต่อเนื่องของอาการจากปลายกิ่งเข้าด้านใน', 'บันทึกต้นที่พบอาการเพื่อสำรวจซ้ำ'],
        preventionTips: ['ตัดแต่งส่วนที่เสียหายตามหลักสุขอนามัยพืชเมื่อจำเป็น', 'ลดความเครียดของต้นด้วยการดูแลน้ำและสภาพแปลงให้เหมาะสม', 'ขอคำแนะนำจากผู้เชี่ยวชาญหากอาการลุกลามหลายกิ่ง'],
    },
    {
        id: 'gall-midge',
        thaiName: 'อาการจากแมลงบั่ว',
        englishName: 'Gall Midge',
        category: 'Pest',
        shortDescription: 'ภาพมีลักษณะใกล้เคียงความผิดปกติจากแมลงบั่วหรือปุ่มนูนบนใบ ควรตรวจใบอ่อนและด้านใต้ใบอย่างละเอียด',
        symptoms: ['ใบมีปุ่มนูน จุดบวม หรือรอยผิดรูป', 'อาการมักสังเกตได้ชัดบนใบอ่อน', 'บางจุดอาจเปลี่ยนสีหรือแห้งตามมา'],
        monitoringTips: ['สำรวจใบอ่อนหลายใบในต้นเดียวกัน', 'ตรวจใต้ใบและบริเวณยอดอ่อน', 'เปรียบเทียบกับใบปกติในต้นเดียวกัน'],
        preventionTips: ['ติดตามความถี่ของอาการเพื่อประเมินแนวโน้ม', 'จัดการเศษใบที่เสียหายอย่างเหมาะสม', 'ปรึกษาเจ้าหน้าที่เกษตรหากพบปุ่มผิดปกติจำนวนมาก'],
    },
    {
        id: 'healthy',
        thaiName: 'ใบปกติ',
        englishName: 'Healthy',
        category: 'Healthy',
        shortDescription: 'ภาพมีลักษณะใกล้เคียงใบมะม่วงปกติ แต่ควรติดตามสุขภาพต้นและสภาพแปลงอย่างสม่ำเสมอ',
        symptoms: ['สีใบสม่ำเสมอ', 'ไม่พบจุดแผลหรือคราบผิดปกติเด่นชัดจากภาพ', 'รูปทรงใบโดยรวมดูสมบูรณ์'],
        monitoringTips: ['สำรวจใบใหม่และใบแก่เป็นระยะ', 'บันทึกภาพเมื่อเริ่มพบจุดหรือรอยผิดปกติ', 'สังเกตความชื้น แสง และการระบายอากาศในทรงพุ่ม'],
        preventionTips: ['ดูแลทรงพุ่มให้โปร่งและสะอาด', 'ให้น้ำและธาตุอาหารตามความเหมาะสมของแปลง', 'ติดตามอาการหลังฝนตกหรือช่วงความชื้นสูง'],
    },
    {
        id: 'powdery-mildew',
        thaiName: 'โรคราแป้ง',
        englishName: 'Powdery Mildew',
        category: 'Disease',
        shortDescription: 'ภาพมีลักษณะใกล้เคียงคราบสีอ่อนหรือผงบนใบ ซึ่งควรตรวจหลายใบและดูสภาพอากาศร่วมด้วย',
        symptoms: ['คราบคล้ายผงสีขาวหรือสีอ่อนบนผิวใบ', 'ใบอ่อนอาจมีรอยผิดรูปหรือชะงักการเจริญ', 'อาการอาจเด่นขึ้นเมื่ออากาศเหมาะกับเชื้อรา'],
        monitoringTips: ['ตรวจทั้งด้านบนและด้านใต้ใบ', 'สำรวจว่าคราบเช็ดออกได้หรือเป็นเนื้อเยื่อเสียหาย', 'ติดตามใบอ่อนและช่ออ่อนในช่วงอากาศชื้น'],
        preventionTips: ['เพิ่มการถ่ายเทอากาศในทรงพุ่ม', 'หลีกเลี่ยงความชื้นสะสมบนใบ', 'ปรึกษาผู้เชี่ยวชาญหากอาการเพิ่มขึ้นรวดเร็ว'],
    },
    {
        id: 'sooty-mould',
        thaiName: 'โรคราดำ',
        englishName: 'Sooty Mould',
        category: 'Disease',
        shortDescription: 'ภาพมีลักษณะใกล้เคียงคราบดำบนใบ ซึ่งอาจเกี่ยวข้องกับแมลงดูดน้ำเลี้ยงและน้ำหวานบนผิวใบ',
        symptoms: ['มีคราบสีดำหรือเทาดำบนผิวใบ', 'คราบอาจปกคลุมเป็นบริเวณกว้าง', 'อาจพบร่วมกับเพลี้ยหรือแมลงที่สร้างน้ำหวาน'],
        monitoringTips: ['ตรวจใต้ใบ กิ่งอ่อน และยอดว่ามีแมลงดูดน้ำเลี้ยงหรือไม่', 'ดูว่าคราบอยู่บนผิวใบหรือเป็นแผลของใบ', 'สำรวจต้นใกล้เคียงเพื่อดูการกระจายของอาการ'],
        preventionTips: ['ลดแหล่งสะสมของแมลงและน้ำหวานบนใบ', 'ดูแลความโปร่งของทรงพุ่ม', 'ขอคำแนะนำจากเจ้าหน้าที่เกษตรหากพบร่วมกับแมลงจำนวนมาก'],
    },
];

const dom = {};

document.addEventListener('DOMContentLoaded', () => {
    cacheDom();
    bindEvents();
    renderHistory();
    resetAnalysisSteps();
    loadModel();
    registerServiceWorker();
});

function cacheDom() {
    [
        'menuButton',
        'topNav',
        'cameraInput',
        'galleryInput',
        'previewShell',
        'previewEmpty',
        'previewImage',
        'scanOverlay',
        'qualityBox',
        'changeImageButton',
        'analyzeButton',
        'analysisSteps',
        'modelStatus',
        'modelDot',
        'retryModelButton',
        'resultEmpty',
        'resultCard',
        'resultStatus',
        'confidenceLevel',
        'confidencePercent',
        'gaugeValue',
        'topThaiName',
        'topEnglishName',
        'lowConfidenceMessage',
        'predictionPanel',
        'predictionList',
        'togglePredictionsButton',
        'infoPanel',
        'shortDescription',
        'symptomsList',
        'monitoringList',
        'preventionList',
        'historyList',
        'clearHistoryButton',
        'historyDialog',
        'historyDialogContent',
        'closeHistoryDialog',
        'stickyAnalyzeBar',
        'stickyAnalyzeButton',
        'toastViewport',
        'confirmDialog',
        'confirmMessage',
        'confirmCancelButton',
        'confirmOkButton',
    ].forEach((id) => {
        dom[id] = document.getElementById(id);
    });
}

function bindEvents() {
    dom.menuButton?.addEventListener('click', toggleMenu);
    dom.topNav?.addEventListener('click', (event) => {
        if (event.target instanceof HTMLAnchorElement) {
            closeMenu();
        }
    });
    dom.cameraInput?.addEventListener('change', handleFileSelection);
    dom.galleryInput?.addEventListener('change', handleFileSelection);
    dom.changeImageButton?.addEventListener('click', () => dom.galleryInput?.click());
    dom.analyzeButton?.addEventListener('click', analyzeCurrentImage);
    dom.stickyAnalyzeButton?.addEventListener('click', analyzeCurrentImage);
    dom.retryModelButton?.addEventListener('click', loadModel);
    dom.togglePredictionsButton?.addEventListener('click', () => {
        showAllPredictions = !showAllPredictions;
        renderPredictionList(latestPredictions);
    });
    dom.clearHistoryButton?.addEventListener('click', clearHistory);
    dom.historyList?.addEventListener('click', handleHistoryClick);
    dom.closeHistoryDialog?.addEventListener('click', () => dom.historyDialog?.close());
    dom.historyDialog?.addEventListener('click', (event) => {
        if (event.target === dom.historyDialog) {
            dom.historyDialog.close();
        }
    });
    dom.confirmCancelButton?.addEventListener('click', () => resolveConfirm(false));
    dom.confirmOkButton?.addEventListener('click', () => resolveConfirm(true));
    dom.confirmDialog?.addEventListener('click', (event) => {
        if (event.target === dom.confirmDialog) {
            resolveConfirm(false);
        }
    });
    initActiveNavigation();
}

function toggleMenu() {
    const isOpen = dom.topNav?.classList.toggle('is-open') || false;
    dom.menuButton?.setAttribute('aria-expanded', String(isOpen));
    dom.menuButton?.setAttribute('aria-label', isOpen ? 'ปิดเมนู' : 'เปิดเมนู');
}

function closeMenu() {
    dom.topNav?.classList.remove('is-open');
    dom.menuButton?.setAttribute('aria-expanded', 'false');
    dom.menuButton?.setAttribute('aria-label', 'เปิดเมนู');
}

async function loadModel() {
    if (modelPromise) {
        return modelPromise;
    }

    setModelState('loading', 'กำลังโหลดโมเดล AI');
    modelPromise = (async () => {
        try {
            if (location.protocol === 'file:') {
                throw new Error('FILE_PROTOCOL_NOT_SUPPORTED');
            }

            await ensureTensorFlowJs();

            model = await tf.loadLayersModel(MODEL_PATH, {
                requestInit: { cache: 'no-store' },
            });
            if (IS_DEVELOPMENT && model.inputs?.[0]?.shape) {
                console.info('Mango Leaf AI model input shape:', model.inputs[0].shape);
            }
            setModelState('ready', 'โมเดลพร้อมใช้งาน');
            updateStickyAnalyzeBar();
            return model;
        } catch (error) {
            model = null;
            modelPromise = null;
            setModelState('error', 'โหลดโมเดลไม่สำเร็จ');
            updateStickyAnalyzeBar();
            showToast('ไม่สามารถโหลดโมเดลได้', 'error');
            showErrorResult('ไม่สามารถโหลดโมเดล AI ได้', getModelLoadHelp(error));
            if (IS_DEVELOPMENT) {
                console.error('Model load failed:', error);
            }
            return null;
        }
    })();

    return modelPromise;
}

async function ensureTensorFlowJs() {
    if (window.tf?.loadLayersModel) {
        return window.tf;
    }

    for (const source of TFJS_SOURCES) {
        try {
            await loadScriptOnce(source);
            if (window.tf?.loadLayersModel) {
                return window.tf;
            }
        } catch (error) {
            if (IS_DEVELOPMENT) {
                console.warn('TensorFlow.js source failed:', source, error);
            }
        }
    }

    throw new Error('TFJS_NOT_AVAILABLE');
}

function loadScriptOnce(src) {
    return new Promise((resolve, reject) => {
        const existingScript = document.querySelector(`script[src="${src}"]`);
        if (existingScript?.dataset.loaded === 'true') {
            resolve();
            return;
        }

        existingScript?.remove();

        const script = document.createElement('script');
        script.src = src;
        script.async = true;
        script.dataset.tfjsFallback = 'true';
        script.onload = () => {
            script.dataset.loaded = 'true';
            resolve();
        };
        script.onerror = () => reject(new Error(`Cannot load ${src}`));

        document.head.appendChild(script);
    });
}

function getModelLoadHelp(error) {
    const message = String(error?.message || error || '');

    if (message.includes('FILE_PROTOCOL_NOT_SUPPORTED')) {
        return 'กรุณาเปิดผ่าน local web server หรือ GitHub Pages แทนการดับเบิลคลิกไฟล์ HTML โดยตรง เพราะเบราว์เซอร์ไม่อนุญาตให้โหลดไฟล์โมเดล .json/.bin ผ่าน file://';
    }

    if (message.includes('TFJS_NOT_AVAILABLE') || message.includes('TensorFlow.js')) {
        return 'ไม่สามารถโหลด TensorFlow.js ได้ กรุณาตรวจสอบอินเทอร์เน็ต หรือรอสักครู่แล้วกด “ลองโหลดโมเดลอีกครั้ง”';
    }

    if (message.includes('404') || message.includes('Not Found') || message.includes('model_mango')) {
        return 'ไม่พบไฟล์โมเดล กรุณาตรวจสอบว่าโฟลเดอร์ model_mango มี model.json และไฟล์ group1-shard1of5.bin ถึง group1-shard5of5.bin อยู่ครบ และตัวพิมพ์ชื่อไฟล์ตรงกัน';
    }

    return 'กรุณาตรวจสอบอินเทอร์เน็ต ล้าง cache ของเว็บ แล้วลองใหม่อีกครั้ง หาก deploy บน GitHub Pages ให้ตรวจว่าโฟลเดอร์ model_mango ถูกอัปโหลดครบทุกไฟล์';
}

function setModelState(state, message) {
    if (dom.modelStatus) {
        dom.modelStatus.textContent = message;
    }
    dom.modelDot?.classList.remove('ready', 'error');
    if (state === 'ready' || state === 'error') {
        dom.modelDot?.classList.add(state);
    }
    dom.retryModelButton?.classList.toggle('hidden', state !== 'error');
}

async function handleFileSelection(event) {
    const input = event.currentTarget;
    const file = input?.files?.[0];
    if (!file) {
        return;
    }

    if (!file.type.startsWith('image/')) {
        showQualityMessages(['ไฟล์นี้ไม่ใช่รูปภาพ กรุณาเลือกไฟล์ JPG, PNG หรือ WebP'], false);
        showToast('ไฟล์นี้ไม่ใช่รูปภาพ', 'warning');
        return;
    }

    currentImageFile = file;
    latestPredictions = [];
    showAllPredictions = false;
    resetResult();
    resetAnalysisSteps();

    try {
        const previewUrl = await createPreviewUrl(file);
        setPreviewImage(previewUrl, file.name || 'ภาพใบมะม่วงที่เลือก');
        await waitForImage(dom.previewImage);
        const quality = await checkImageQuality(dom.previewImage);
        showQualityMessages(quality.messages, quality.ok);
        dom.changeImageButton.disabled = false;
        dom.analyzeButton.disabled = false;
        updateStickyAnalyzeBar();
        showToast('เลือกภาพสำเร็จ', 'success');
    } catch (error) {
        showQualityMessages(['ไม่สามารถอ่านภาพนี้ได้ กรุณาเลือกภาพใหม่'], false);
        updateStickyAnalyzeBar();
        showToast('ไม่สามารถอ่านภาพนี้ได้', 'error');
        if (IS_DEVELOPMENT) {
            console.error('Image preview failed:', error);
        }
    } finally {
        input.value = '';
    }
}

async function createPreviewUrl(file) {
    if (currentImageUrl) {
        URL.revokeObjectURL(currentImageUrl);
        currentImageUrl = '';
    }

    if ('createImageBitmap' in window) {
        try {
            const bitmap = await createImageBitmap(file, { imageOrientation: 'from-image' });
            const canvas = document.createElement('canvas');
            const scale = Math.min(1, 1600 / Math.max(bitmap.width, bitmap.height));
            canvas.width = Math.max(1, Math.round(bitmap.width * scale));
            canvas.height = Math.max(1, Math.round(bitmap.height * scale));
            const context = canvas.getContext('2d');
            if (!context) {
                throw new Error('Canvas context is not available');
            }
            context.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
            bitmap.close?.();
            const blob = await new Promise((resolve) => canvas.toBlob(resolve, 'image/jpeg', 0.9));
            if (blob) {
                currentImageUrl = URL.createObjectURL(blob);
                return currentImageUrl;
            }
        } catch (error) {
            if (IS_DEVELOPMENT) {
                console.warn('createImageBitmap fallback:', error);
            }
        }
    }

    currentImageUrl = URL.createObjectURL(file);
    return currentImageUrl;
}

function setPreviewImage(url, altText) {
    if (!dom.previewImage || !dom.previewEmpty) {
        return;
    }

    dom.previewImage.src = url;
    dom.previewImage.alt = altText;
    dom.previewImage.classList.remove('hidden');
    dom.previewEmpty.classList.add('hidden');
}

function waitForImage(image) {
    if (!image) {
        return Promise.reject(new Error('Preview image element is missing'));
    }
    if (image.complete && image.naturalWidth > 0) {
        return Promise.resolve();
    }
    return new Promise((resolve, reject) => {
        image.onload = () => resolve();
        image.onerror = () => reject(new Error('Image failed to load'));
    });
}

async function checkImageQuality(image) {
    const messages = [];
    const width = image.naturalWidth || image.width;
    const height = image.naturalHeight || image.height;

    if (width < 320 || height < 320) {
        messages.push('ภาพมีขนาดค่อนข้างเล็ก กรุณาใช้ภาพที่เห็นใบมะม่วงชัดเจนกว่านี้');
    }

    const sample = drawImageSample(image, 160);
    const metrics = calculateImageMetrics(sample.imageData);

    if (metrics.brightness < 58) {
        messages.push('ภาพค่อนข้างมืด กรุณาถ่ายในบริเวณที่มีแสงเพียงพอ');
    } else if (metrics.brightness > 218) {
        messages.push('ภาพค่อนข้างสว่างเกินไป กรุณาหลีกเลี่ยงแสงจ้าหรือเงาสะท้อนบนใบ');
    }

    if (metrics.blurScore < 75) {
        messages.push('ภาพอาจไม่ชัดเจน กรุณาถ่ายใบมะม่วงให้เต็มกรอบและถือกล้องให้นิ่ง');
    }

    if (messages.length === 0) {
        messages.push('คุณภาพภาพพร้อมสำหรับการวิเคราะห์');
    }

    return {
        ok: messages.length === 1 && messages[0].startsWith('คุณภาพภาพ'),
        messages,
    };
}

function drawImageSample(image, maxSize) {
    const ratio = Math.min(maxSize / image.naturalWidth, maxSize / image.naturalHeight, 1);
    const width = Math.max(1, Math.round(image.naturalWidth * ratio));
    const height = Math.max(1, Math.round(image.naturalHeight * ratio));
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const context = canvas.getContext('2d', { willReadFrequently: true });
    context.drawImage(image, 0, 0, width, height);
    return {
        width,
        height,
        imageData: context.getImageData(0, 0, width, height),
    };
}

function calculateImageMetrics(imageData) {
    const { data, width, height } = imageData;
    const grayscale = new Float32Array(width * height);
    let brightnessTotal = 0;

    for (let i = 0, pixel = 0; i < data.length; i += 4, pixel += 1) {
        const gray = data[i] * 0.299 + data[i + 1] * 0.587 + data[i + 2] * 0.114;
        grayscale[pixel] = gray;
        brightnessTotal += gray;
    }

    const brightness = brightnessTotal / grayscale.length;
    const laplacianValues = [];

    for (let y = 1; y < height - 1; y += 1) {
        for (let x = 1; x < width - 1; x += 1) {
            const center = grayscale[y * width + x] * 4;
            const neighbors =
                grayscale[(y - 1) * width + x] +
                grayscale[(y + 1) * width + x] +
                grayscale[y * width + x - 1] +
                grayscale[y * width + x + 1];
            laplacianValues.push(center - neighbors);
        }
    }

    const blurScore = variance(laplacianValues);
    return { brightness, blurScore };
}

function variance(values) {
    if (!values.length) {
        return 0;
    }
    const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
    return values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length;
}

function showQualityMessages(messages, isGood) {
    if (!dom.qualityBox) {
        return;
    }
    dom.qualityBox.innerHTML = messages.map((message) => `<p>${escapeHtml(message)}</p>`).join('');
    dom.qualityBox.classList.toggle('good', isGood);
    dom.qualityBox.classList.remove('hidden');
}

async function analyzeCurrentImage() {
    if (isPredicting || !dom.previewImage || dom.previewImage.classList.contains('hidden')) {
        showToast('กรุณาเลือกหรือถ่ายภาพก่อนวิเคราะห์', 'warning');
        return;
    }

    const loadedModel = model || await loadModel();
    if (!loadedModel) {
        return;
    }

    isPredicting = true;
    dom.analyzeButton.disabled = true;
    dom.changeImageButton.disabled = true;
    if (dom.stickyAnalyzeButton) {
        dom.stickyAnalyzeButton.disabled = true;
    }
    updateStickyAnalyzeBar();
    dom.scanOverlay?.classList.remove('hidden');
    resetAnalysisSteps();

    try {
        setAnalysisStep('prepare');
        await nextFrame();
        await checkImageQuality(dom.previewImage);

        setAnalysisStep('scan');
        await nextFrame();

        const inputTensor = tf.tidy(() => {
            const pixels = tf.browser.fromPixels(dom.previewImage, 3);
            const resized = tf.image.resizeBilinear(pixels, INPUT_SIZE);
            // Notebook training used ImageDataGenerator(rescale=1./255) with RGB 224x224 input.
            return resized.toFloat().div(255).expandDims(0);
        });

        const predictionTensor = loadedModel.predict(inputTensor);
        const outputTensor = Array.isArray(predictionTensor) ? predictionTensor[0] : predictionTensor;
        const rawPredictions = Array.from(await outputTensor.data());

        inputTensor.dispose();
        tf.dispose(predictionTensor);

        setAnalysisStep('score');
        await nextFrame();
        latestPredictions = formatPredictions(rawPredictions);
        renderResult(latestPredictions);
        saveHistory(latestPredictions);
        renderHistory();
        setAnalysisStep('done');
        showToast('วิเคราะห์เสร็จแล้ว', 'success');
        showToast('บันทึกประวัติแล้ว', 'success');
    } catch (error) {
        showErrorResult('วิเคราะห์ภาพไม่สำเร็จ', 'โปรดลองเลือกภาพใหม่หรือโหลดหน้าเว็บอีกครั้ง หากยังพบปัญหาให้ตรวจสอบไฟล์โมเดลและ TensorFlow.js');
        setModelState(model ? 'ready' : 'error', model ? 'โมเดลพร้อมใช้งาน' : 'โหลดโมเดลไม่สำเร็จ');
        showToast('วิเคราะห์ภาพไม่สำเร็จ', 'error');
        if (IS_DEVELOPMENT) {
            console.error('Prediction failed:', error);
        }
    } finally {
        isPredicting = false;
        dom.scanOverlay?.classList.add('hidden');
        dom.analyzeButton.disabled = false;
        dom.changeImageButton.disabled = false;
        if (dom.stickyAnalyzeButton) {
            dom.stickyAnalyzeButton.disabled = false;
        }
        updateStickyAnalyzeBar();
    }
}

function formatPredictions(predictions) {
    return predictions
        .map((probability, index) => ({
            ...classes[index],
            probability,
            percent: probability * 100,
        }))
        .sort((a, b) => b.probability - a.probability);
}

function renderResult(predictions) {
    const top = predictions[0];
    const confidence = getConfidenceLevel(top.probability);
    const isHealthy = top.category === 'Healthy';
    const isLowConfidence = top.probability < LOW_CONFIDENCE_THRESHOLD;

    dom.resultEmpty?.classList.add('hidden');
    dom.resultCard?.classList.remove('hidden');
    dom.predictionPanel?.classList.remove('hidden');
    dom.infoPanel?.classList.remove('hidden');
    dom.lowConfidenceMessage?.classList.toggle('hidden', !isLowConfidence);

    if (dom.resultStatus) {
        dom.resultStatus.textContent = isLowConfidence ? 'ควรถ่ายภาพใหม่' : (isHealthy ? 'สถานะปกติ' : 'พบความเสี่ยงของอาการ');
        dom.resultStatus.className = `status-badge ${isLowConfidence ? 'low' : (isHealthy ? '' : 'risk')}`;
    }
    if (dom.confidenceLevel) {
        dom.confidenceLevel.textContent = `ระดับความมั่นใจ: ${confidence.label}`;
    }
    if (dom.confidencePercent) {
        dom.confidencePercent.textContent = `${top.percent.toFixed(2)}%`;
    }
    updateGauge(top.probability, confidence.state);

    if (dom.topThaiName) {
        dom.topThaiName.textContent = top.thaiName;
    }
    if (dom.topEnglishName) {
        dom.topEnglishName.textContent = top.englishName;
    }

    renderPredictionList(predictions);
    renderInfoPanel(top);
}

function getConfidenceLevel(probability) {
    if (probability >= HIGH_CONFIDENCE_THRESHOLD) {
        return { label: 'สูง', state: 'high' };
    }
    if (probability >= LOW_CONFIDENCE_THRESHOLD) {
        return { label: 'ปานกลาง', state: 'medium' };
    }
    return { label: 'ต่ำ', state: 'low' };
}

function updateGauge(probability, state) {
    const circumference = 2 * Math.PI * 48;
    const offset = circumference * (1 - Math.max(0, Math.min(probability, 1)));
    if (dom.gaugeValue) {
        dom.gaugeValue.style.strokeDashoffset = String(offset);
        dom.gaugeValue.classList.remove('medium', 'low');
        if (state === 'medium' || state === 'low') {
            dom.gaugeValue.classList.add(state);
        }
    }
}

function renderPredictionList(predictions) {
    if (!dom.predictionList || !dom.togglePredictionsButton || !predictions.length) {
        return;
    }

    const visiblePredictions = showAllPredictions ? predictions : predictions.slice(0, 3);
    dom.predictionList.innerHTML = '';
    visiblePredictions.forEach((prediction) => {
        const item = document.createElement('div');
        item.className = 'prediction-item';
        item.innerHTML = `
            <div class="prediction-name">
                <strong>${escapeHtml(prediction.thaiName)}</strong>
                <span>${escapeHtml(prediction.englishName)}</span>
            </div>
            <div class="prediction-percent">${prediction.percent.toFixed(2)}%</div>
            <div class="prediction-bar" aria-hidden="true">
                <span style="--value: ${Math.max(prediction.percent, 1).toFixed(2)}%"></span>
            </div>
        `;
        dom.predictionList.appendChild(item);
    });

    dom.togglePredictionsButton.textContent = showAllPredictions ? 'แสดงเฉพาะ Top 3' : 'ดูผลทั้งหมด 8 รายการ';
}

function renderInfoPanel(info) {
    if (!dom.infoPanel) {
        return;
    }
    dom.shortDescription.textContent = info.shortDescription;
    renderList(dom.symptomsList, info.symptoms);
    renderList(dom.monitoringList, info.monitoringTips);
    renderList(dom.preventionList, info.preventionTips);
}

function renderList(element, items) {
    if (!element) {
        return;
    }
    element.innerHTML = '';
    items.forEach((item) => {
        const li = document.createElement('li');
        li.textContent = item;
        element.appendChild(li);
    });
}

function resetResult() {
    if (dom.resultEmpty) {
        dom.resultEmpty.innerHTML = DEFAULT_RESULT_EMPTY_HTML;
    }
    dom.resultEmpty?.classList.remove('hidden');
    dom.resultCard?.classList.add('hidden');
    dom.predictionPanel?.classList.add('hidden');
    dom.infoPanel?.classList.add('hidden');
    dom.lowConfidenceMessage?.classList.add('hidden');
}

function showErrorResult(title, detail) {
    dom.resultEmpty?.classList.remove('hidden');
    dom.resultCard?.classList.add('hidden');
    dom.predictionPanel?.classList.add('hidden');
    dom.infoPanel?.classList.add('hidden');
    if (dom.resultEmpty) {
        dom.resultEmpty.innerHTML = `
            <p class="eyebrow">Error</p>
            <h2>${escapeHtml(title)}</h2>
            <p>${escapeHtml(detail)}</p>
        `;
    }
}

function resetAnalysisSteps() {
    dom.analysisSteps?.querySelectorAll('span').forEach((step) => {
        step.classList.remove('active', 'done');
    });
}

function setAnalysisStep(activeStep) {
    let passedActive = false;
    dom.analysisSteps?.querySelectorAll('span').forEach((step) => {
        const isActive = step.dataset.step === activeStep;
        step.classList.toggle('active', isActive);
        if (isActive) {
            passedActive = true;
        } else if (!passedActive) {
            step.classList.add('done');
        }
    });
}

function nextFrame() {
    return new Promise((resolve) => requestAnimationFrame(() => resolve()));
}

function getHistory() {
    try {
        return JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]');
    } catch {
        return [];
    }
}

function setHistory(items) {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(items.slice(0, 30)));
}

function saveHistory(predictions) {
    if (!predictions.length) {
        return;
    }
    const top = predictions[0];
    const historyItem = {
        id: `${Date.now()}-${Math.random().toString(16).slice(2)}`,
        timestamp: new Date().toISOString(),
        topClass: {
            thaiName: top.thaiName,
            englishName: top.englishName,
            category: top.category,
        },
        confidence: top.probability,
        top3: predictions.slice(0, 3).map((prediction) => ({
            thaiName: prediction.thaiName,
            englishName: prediction.englishName,
            probability: prediction.probability,
        })),
    };
    setHistory([historyItem, ...getHistory()]);
}

function renderHistory() {
    if (!dom.historyList) {
        return;
    }

    const history = getHistory();
    dom.clearHistoryButton.disabled = history.length === 0;

    if (history.length === 0) {
        dom.historyList.innerHTML = '<div class="history-empty">ยังไม่มีประวัติการตรวจ</div>';
        return;
    }

    dom.historyList.innerHTML = '';
    history.forEach((item) => {
        const date = new Date(item.timestamp);
        const element = document.createElement('article');
        element.className = 'history-item';
        element.dataset.id = item.id;
        element.innerHTML = `
            <div>
                <strong>${escapeHtml(item.topClass.thaiName)} (${escapeHtml(item.topClass.englishName)})</strong>
                <span>${formatDate(date)} เวลา ${formatTime(date)}</span>
                <span>ความมั่นใจ ${(item.confidence * 100).toFixed(2)}%</span>
            </div>
            <div class="history-actions">
                <button class="button ghost" type="button" data-action="detail">ดูรายละเอียด</button>
                <button class="button danger" type="button" data-action="delete">ลบรายการ</button>
            </div>
        `;
        dom.historyList.appendChild(element);
    });
}

function handleHistoryClick(event) {
    if (!(event.target instanceof Element)) {
        return;
    }
    const button = event.target.closest('button[data-action]');
    const itemElement = event.target.closest('.history-item');
    if (!button || !itemElement) {
        return;
    }

    const id = itemElement.dataset.id;
    const action = button.dataset.action;
    if (action === 'delete') {
        deleteHistoryItem(id);
    } else if (action === 'detail') {
        showHistoryDetail(id);
    }
}

function deleteHistoryItem(id) {
    setHistory(getHistory().filter((item) => item.id !== id));
    renderHistory();
}

function clearHistory() {
    if (!getHistory().length) {
        return;
    }
    const confirmed = window.confirm('ต้องการล้างประวัติการตรวจทั้งหมดหรือไม่');
    if (!confirmed) {
        return;
    }
    setHistory([]);
    renderHistory();
}

function showHistoryDetail(id) {
    const item = getHistory().find((historyItem) => historyItem.id === id);
    if (!item || !dom.historyDialogContent || !dom.historyDialog) {
        return;
    }
    const date = new Date(item.timestamp);
    dom.historyDialogContent.innerHTML = `
        <p><strong>วันที่:</strong> ${formatDate(date)} เวลา ${formatTime(date)}</p>
        <p><strong>ผลการประเมิน:</strong> ${escapeHtml(item.topClass.thaiName)} (${escapeHtml(item.topClass.englishName)})</p>
        <p><strong>ความมั่นใจ:</strong> ${(item.confidence * 100).toFixed(2)}%</p>
        <div class="dialog-predictions">
            ${item.top3.map((prediction) => `
                <div class="prediction-item">
                    <div class="prediction-name">
                        <strong>${escapeHtml(prediction.thaiName)}</strong>
                        <span>${escapeHtml(prediction.englishName)}</span>
                    </div>
                    <div class="prediction-percent">${(prediction.probability * 100).toFixed(2)}%</div>
                    <div class="prediction-bar" aria-hidden="true">
                        <span style="--value: ${Math.max(prediction.probability * 100, 1).toFixed(2)}%"></span>
                    </div>
                </div>
            `).join('')}
        </div>
    `;
    dom.historyDialog.showModal();
}

function formatDate(date) {
    return new Intl.DateTimeFormat('th-TH', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
    }).format(date);
}

function formatTime(date) {
    return new Intl.DateTimeFormat('th-TH', {
        hour: '2-digit',
        minute: '2-digit',
    }).format(date);
}

function escapeHtml(value) {
    return String(value)
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#039;');
}

function updateStickyAnalyzeBar() {
    if (!dom.stickyAnalyzeBar || !dom.stickyAnalyzeButton) {
        return;
    }

    const hasImage = Boolean(dom.previewImage && !dom.previewImage.classList.contains('hidden'));
    dom.stickyAnalyzeBar.classList.toggle('hidden', !hasImage || isPredicting);
    dom.stickyAnalyzeButton.disabled = !hasImage || isPredicting || !model;
}

function showToast(message, type = 'success') {
    if (!dom.toastViewport) {
        return;
    }

    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    dom.toastViewport.appendChild(toast);

    window.setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateY(10px) scale(0.98)';
        window.setTimeout(() => toast.remove(), 180);
    }, 2600);
}

function requestConfirm(message) {
    if (!dom.confirmDialog || !dom.confirmMessage) {
        showToast('ไม่พบกล่องยืนยัน กรุณาลองใหม่', 'error');
        return Promise.resolve(false);
    }

    dom.confirmMessage.textContent = message;
    dom.confirmDialog.showModal();

    return new Promise((resolve) => {
        pendingConfirmResolve = resolve;
    });
}

function resolveConfirm(value) {
    if (dom.confirmDialog?.open) {
        dom.confirmDialog.close();
    }

    if (pendingConfirmResolve) {
        pendingConfirmResolve(value);
        pendingConfirmResolve = null;
    }
}

function initActiveNavigation() {
    const navLinks = Array.from(document.querySelectorAll('.bottom-nav a[data-nav-target]'));
    const sections = ['home', 'scanner', 'history', 'guide']
        .map((id) => document.getElementById(id))
        .filter(Boolean);

    if (!navLinks.length || !sections.length || !('IntersectionObserver' in window)) {
        return;
    }

    const setActive = (id) => {
        navLinks.forEach((link) => {
            link.classList.toggle('active', link.dataset.navTarget === id);
        });
    };

    const observer = new IntersectionObserver((entries) => {
        const visible = entries
            .filter((entry) => entry.isIntersecting)
            .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];

        if (visible?.target?.id) {
            setActive(visible.target.id);
        }
    }, {
        rootMargin: '-30% 0px -55% 0px',
        threshold: [0.12, 0.25, 0.5],
    });

    sections.forEach((section) => observer.observe(section));
    setActive('home');
}

function renderHistory() {
    if (!dom.historyList) {
        return;
    }

    const history = getHistory();
    dom.clearHistoryButton.disabled = history.length === 0;

    if (history.length === 0) {
        dom.historyList.innerHTML = `
            <div class="history-empty">
                <span class="empty-leaf" aria-hidden="true"></span>
                <strong>ยังไม่มีประวัติการตรวจ</strong>
                <p>เมื่อคุณวิเคราะห์ภาพใบมะม่วง ผลลัพธ์จะถูกแสดงไว้ที่นี่</p>
                <a class="button secondary" href="#scanner">เริ่มตรวจใบมะม่วง</a>
            </div>
        `;
        return;
    }

    dom.historyList.innerHTML = '';
    history.forEach((item) => {
        const date = new Date(item.timestamp);
        const element = document.createElement('article');
        element.className = 'history-item';
        element.dataset.id = item.id;
        element.innerHTML = `
            <div>
                <span>${formatDate(date)} • ${formatTime(date)}</span>
                <strong>${escapeHtml(item.topClass.thaiName)}</strong>
                <span>${escapeHtml(item.topClass.englishName)} • ความมั่นใจ ${(item.confidence * 100).toFixed(2)}%</span>
            </div>
            <div class="history-actions">
                <button class="button ghost" type="button" data-action="detail">ดูรายละเอียด</button>
                <button class="button danger" type="button" data-action="delete">ลบรายการ</button>
            </div>
        `;
        dom.historyList.appendChild(element);
    });
}

function deleteHistoryItem(id) {
    setHistory(getHistory().filter((item) => item.id !== id));
    renderHistory();
    showToast('ลบรายการแล้ว', 'success');
}

async function clearHistory() {
    if (!getHistory().length) {
        return;
    }

    const confirmed = await requestConfirm('ต้องการล้างประวัติการตรวจทั้งหมดหรือไม่');
    if (!confirmed) {
        return;
    }

    setHistory([]);
    renderHistory();
    showToast('ล้างประวัติแล้ว', 'success');
}

function registerServiceWorker() {
    if (!('serviceWorker' in navigator) || location.protocol === 'file:') {
        return;
    }
    navigator.serviceWorker.register('./service-worker.js').catch((error) => {
        if (IS_DEVELOPMENT) {
            console.warn('Service worker registration failed:', error);
        }
    });
}

window.addEventListener('beforeunload', () => {
    if (currentImageUrl) {
        URL.revokeObjectURL(currentImageUrl);
    }
});
