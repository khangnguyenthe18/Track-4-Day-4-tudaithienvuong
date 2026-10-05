// Client-side Application Logic - 100% Tiếng Việt cho Hệ thống Giám Sát Người Đi Bộ & Sức Khỏe Camera

document.addEventListener('DOMContentLoaded', () => {
  // DOM Elements
  const systemStatus = document.getElementById('systemStatus');
  const fpsVal = document.getElementById('fpsVal');
  const latencyVal = document.getElementById('latencyVal');
  const frameCounter = document.getElementById('frameCounter');
  const decisionHUD = document.getElementById('decisionHUD');
  const hudDecisionText = document.getElementById('hudDecisionText');
  const hudConfidence = document.getElementById('hudConfidence');

  const btnTogglePlay = document.getElementById('btnTogglePlay');
  const playText = document.getElementById('playText');
  const btnReset = document.getElementById('btnReset');
  const btnExport = document.getElementById('btnExport');
  const btnUploadTrigger = document.getElementById('btnUploadTrigger');

  // Source & Upload Elements
  const sourceBadge = document.getElementById('sourceBadge');
  const watermarkSource = document.getElementById('watermarkSource');
  const scenarioBtns = document.querySelectorAll('.scenario-btn');
  const videoFileInput = document.getElementById('videoFileInput');
  const btnSelectFile = document.getElementById('btnSelectFile');
  const uploadDropzone = document.getElementById('uploadDropzone');
  const uploadStatusText = document.getElementById('uploadStatusText');
  const uploadedFilesWrapper = document.getElementById('uploadedFilesWrapper');
  const uploadedChipsContainer = document.getElementById('uploadedChipsContainer');

  // Tracking Table Elements
  const tracksTableBody = document.getElementById('tracksTableBody');
  const activeTrackCount = document.getElementById('activeTrackCount');

  // Health Elements
  const overallCircleProgress = document.getElementById('overallCircleProgress');
  const overallGaugeText = document.getElementById('overallGaugeText');
  const healthScoreBadge = document.getElementById('healthScoreBadge');
  const valBlur = document.getElementById('valBlur');
  const barBlur = document.getElementById('barBlur');
  const valBright = document.getElementById('valBright');
  const barBright = document.getElementById('barBright');
  const valOccl = document.getElementById('valOccl');
  const barOccl = document.getElementById('barOccl');
  const valNoise = document.getElementById('valNoise');
  const barNoise = document.getElementById('barNoise');
  const valFreeze = document.getElementById('valFreeze');
  const barFreeze = document.getElementById('barFreeze');
  const healthTagsContainer = document.getElementById('healthTagsContainer');

  // Probabilities & Decision
  const probNormalVal = document.getElementById('probNormalVal');
  const probNormalFill = document.getElementById('probNormalFill');
  const probRareVal = document.getElementById('probRareVal');
  const probRareFill = document.getElementById('probRareFill');
  const probArtifactVal = document.getElementById('probArtifactVal');
  const probArtifactFill = document.getElementById('probArtifactFill');
  const logFeed = document.getElementById('logFeed');

  let isPlaying = true;
  let pollInterval = null;

  const SCENARIO_LABELS = {
    normal_day: 'Ban ngày',
    night_rain_rare: 'Đêm mưa',
    dirty_lens_artifact: 'Ống kính bẩn',
    frozen_camera: 'Đứng hình'
  };

  // Từ điển dịch nhãn quyết định
  const DECISION_LABELS = {
    "NORMAL": "Bình thường",
    "REAL_RARE_EVENT": "Người đi bộ trong điều kiện khó",
    "CAMERA_ARTIFACT": "Báo động giả / Vết bẩn",
    "CAMERA_FAULT": "Lỗi camera",
    "NO_PEDESTRIAN": "Không có người đi bộ",
    "INITIALIZING": "Đang khởi tạo"
  };

  // Từ điển dịch nhãn chuẩn đoán sức khỏe camera
  const TAG_TRANSLATIONS = {
    "NORMAL_HEALTH": "CAMERA_TỐT",
    "SHARP": "ẢNH_SẮC_NÉT",
    "MODERATE_BLUR": "ẢNH_HƠI_MỜ",
    "HEAVY_BLUR": "ẢNH_MỜ_NẶNG",
    "OPTIMAL": "ÁNH_SÁNG_TỐT",
    "LOW_LIGHT": "ÁNH_SÁNG_YẾU",
    "VERY_DARK": "THIẾU_SÁNG_TRẦM_TRỌNG",
    "OVEREXPOSED": "CHÓI_SÁNG_LÓA",
    "SLIGHTLY_OVEREXPOSED": "HƠI_CHÓI",
    "CLEAR": "LENS_SẠCH",
    "PARTIAL_SMUDGE": "VẾT_BẨN_ỐNG_KÍNH",
    "HEAVY_OCCLUSION": "BỊ_CHE_KHUẤT_NẶNG",
    "CLEAN": "CẢM_BIẾN_SẠCH",
    "SLIGHT_NOISE": "NHIỄU_HẠT_NHẸ",
    "HEAVY_NOISE": "NHIỄU_HẠT_NẶNG",
    "ACTIVE": "TRUYỀN_MƯỢT_MÀ",
    "POSSIBLY_STATIC": "TĨNH_BẤT_THƯỜNG",
    "FROZEN": "CAMERA_ĐỨNG_HÌNH"
  };

  // 1. Tương tác tải Video lên (Upload Video)
  function triggerFileSelect() {
    videoFileInput.click();
  }

  btnUploadTrigger.addEventListener('click', triggerFileSelect);
  btnSelectFile.addEventListener('click', triggerFileSelect);

  videoFileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFileUpload(e.target.files[0]);
    }
  });

  // Drag and drop events
  ['dragenter', 'dragover'].forEach(eventName => {
    uploadDropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      uploadDropzone.classList.add('dragover');
    });
  });

  ['dragleave', 'drop'].forEach(eventName => {
    uploadDropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      uploadDropzone.classList.remove('dragover');
    });
  });

  uploadDropzone.addEventListener('drop', (e) => {
    const dt = e.dataTransfer;
    if (dt && dt.files && dt.files.length > 0) {
      handleFileUpload(dt.files[0]);
    }
  });

  async function handleFileUpload(file) {
    if (!file) return;

    uploadStatusText.innerText = `Đang tải lên: ${file.name} (${(file.size / (1024*1024)).toFixed(1)} MB)...`;
    uploadStatusText.style.color = "var(--accent-blue)";

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch('/api/upload-video', {
        method: 'POST',
        body: formData
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || 'Không thể tải tệp lên');
      }

      const data = await res.json();
      uploadStatusText.innerText = `Đang phân tích video thực tế: ${data.filename} (${data.total_frames} khung hình)`;
      uploadStatusText.style.color = "var(--color-normal)";

      await fetchUploadedVideosList();
      await fetchTelemetry();
    } catch (err) {
      console.error("Lỗi upload video:", err);
      uploadStatusText.innerText = `Lỗi: ${err.message}`;
      uploadStatusText.style.color = "var(--color-artifact)";
    }
  }

  async function fetchUploadedVideosList() {
    try {
      const res = await fetch('/api/uploaded-videos');
      if (!res.ok) return;
      const data = await res.json();
      const videos = data.videos || [];

      if (videos.length === 0) {
        uploadedFilesWrapper.style.display = 'none';
        return;
      }

      uploadedFilesWrapper.style.display = 'block';
      uploadedChipsContainer.innerHTML = videos.map(v => {
        const activeCls = v.is_active ? 'active' : '';
        return `
          <button type="button" class="video-chip ${activeCls}" data-filename="${v.filename}">
            <span>📹 ${v.filename}</span>
            <span class="chip-meta">${v.size_mb}MB</span>
          </button>
        `;
      }).join('');

      // Add click listener to chips
      uploadedChipsContainer.querySelectorAll('.video-chip').forEach(chip => {
        chip.addEventListener('click', async () => {
          const filename = chip.dataset.filename;
          try {
            const resp = await fetch('/api/select-uploaded-video', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ filename })
            });
            if (!resp.ok) throw new Error('Không thể kích hoạt video');
            uploadStatusText.innerText = `Đang phân tích video thực tế: ${filename}`;
            uploadStatusText.style.color = "var(--color-normal)";
            await fetchUploadedVideosList();
            await fetchTelemetry();
          } catch (err) {
            alert('Lỗi chọn video: ' + err.message);
          }
        });
      });
    } catch (err) {
      console.warn("Lỗi tải danh sách video đã upload:", err);
    }
  }

  // 2. Thiết lập sự kiện nút bấm tình huống mô phỏng
  scenarioBtns.forEach(btn => {
    btn.addEventListener('click', async () => {
      const scenario = btn.getAttribute('data-scenario');
      try {
        const response = await fetch('/api/set-scenario', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ scenario })
        });
        if (!response.ok) throw new Error('Không thể chuyển tình huống');
        await fetchUploadedVideosList();
        await fetchTelemetry();
      } catch (err) {
        console.error("Lỗi chuyển kịch bản:", err);
      }
    });
  });

  // 3. Tạm dừng / Tiếp tục
  btnTogglePlay.addEventListener('click', async () => {
    try {
      const res = await fetch('/api/toggle-playback', { method: 'POST' });
      const data = await res.json();
      isPlaying = data.is_playing;
      playText.innerText = isPlaying ? "Tạm dừng" : "Tiếp tục";
      await fetchTelemetry();
    } catch (err) {
      console.error("Lỗi toggle phát:", err);
    }
  });

  // 4. Đặt lại luồng
  btnReset.addEventListener('click', async () => {
    try {
      await fetch('/api/reset', { method: 'POST' });
    } catch (err) {
      console.error("Lỗi reset:", err);
    }
  });

  // 5. Xuất Video MP4
  btnExport.addEventListener('click', async () => {
    btnExport.disabled = true;
    btnExport.innerText = "Đang xuất video…";
    try {
      const res = await fetch('/api/generate-all-videos', { method: 'POST' });
      if (!res.ok) throw new Error('Không thể xuất video');
      await res.json();
      alert("Đã xuất 4 video vào thư mục data/synthetic_videos/.");
    } catch (err) {
      alert("Xuất video thất bại: " + err);
    } finally {
      btnExport.innerText = "Xuất 4 video MP4";
      btnExport.disabled = false;
    }
  });

  // 6. Vòng lặp lấy dữ liệu đo từ xa (Telemetry Polling)
  async function fetchTelemetry() {
    try {
      const response = await fetch('/api/telemetry');
      if (!response.ok) return;
      const data = await response.json();
      updateDashboard(data);
    } catch (err) {
      console.warn("Lỗi tải telemetry:", err);
    }
  }

  function getBarClass(val) {
    if (val >= 0.70) return 'fill-good';
    if (val >= 0.40) return 'fill-degraded';
    return 'fill-red';
  }

  function updateDashboard(data) {
    if (!data) return;

    if (data.is_playing !== undefined) {
      isPlaying = data.is_playing;
      playText.innerText = isPlaying ? 'Tạm dừng' : 'Tiếp tục';
      systemStatus.innerText = isPlaying ? 'Đang chạy' : 'Tạm dừng';
      systemStatus.classList.toggle('is-paused', !isPlaying);
    }

    // Source Mode UI & Watermark
    if (data.source_mode === 'uploaded') {
      sourceBadge.className = 'source-mode-badge badge-uploaded';
      sourceBadge.innerText = `Video thực tế: ${data.video_name || ''}`;
      watermarkSource.innerText = `[Video thực tế: ${data.video_name || ''}]`;
      scenarioBtns.forEach(btn => {
        btn.classList.remove('active');
        btn.setAttribute('aria-pressed', 'false');
      });
    } else {
      sourceBadge.className = 'source-mode-badge badge-synthetic';
      const scName = SCENARIO_LABELS[data.scenario] || data.scenario;
      sourceBadge.innerText = `Mô phỏng: ${scName}`;
      watermarkSource.innerText = `[Mô phỏng: ${scName}]`;

      scenarioBtns.forEach(btn => {
        const selected = btn.dataset.scenario === data.scenario;
        btn.classList.toggle('active', selected);
        btn.setAttribute('aria-pressed', String(selected));
      });
    }

    // Cập nhật Header
    if (data.fps !== undefined) fpsVal.innerText = data.fps;
    if (data.latency_ms !== undefined) latencyVal.innerText = `${data.latency_ms} ms`;
    if (data.frame_idx !== undefined) {
      const totalStr = data.total_frames ? `/${data.total_frames}` : '';
      frameCounter.innerText = `Khung hình #${data.frame_idx}${totalStr}`;
    }

    // Cập nhật HUD Quyết Định Tối Cao
    const decisionKey = data.primary_decision || "INITIALIZING";
    const decisionVN = DECISION_LABELS[decisionKey] || decisionKey.replace(/_/g, ' ');
    const conf = data.primary_confidence !== undefined ? Math.round(data.primary_confidence * 100) : 0;
    
    hudDecisionText.innerText = decisionVN;
    hudConfidence.innerText = `Độ tin cậy: ${conf}%`;

    decisionHUD.className = 'decision-hud';
    if (['NORMAL', 'NO_PEDESTRIAN', 'INITIALIZING'].includes(decisionKey)) {
      decisionHUD.classList.add('badge-normal');
    } else if (decisionKey === 'REAL_RARE_EVENT') {
      decisionHUD.classList.add('badge-rare');
    } else {
      decisionHUD.classList.add('badge-artifact');
    }

    // Cập nhật 💡 Explainable AI (XAI)
    const explainableCard = document.getElementById('explainableCard');
    const explainableText = document.getElementById('explainableText');
    if (explainableText) {
      const expStr = data.primary_explanation 
        || (data.predictions && data.predictions.length > 0 ? data.predictions[0].explanation : null)
        || "Đang phân tích khung hình...";
      explainableText.innerText = expStr;

      if (explainableCard) {
        explainableCard.className = 'explainable-card';
        if (['CAMERA_ARTIFACT', 'CAMERA_FAULT'].includes(decisionKey)) {
          explainableCard.classList.add('border-artifact');
        } else if (decisionKey === 'REAL_RARE_EVENT') {
          explainableCard.classList.add('border-rare');
        } else {
          explainableCard.classList.add('border-normal');
        }
      }
    }

    // Cập nhật Vector Sức Khỏe Camera
    if (data.camera_health) {
      const ch = data.camera_health;
      const pct = Math.round(ch.overall_health * 100);
      overallGaugeText.textContent = `${pct}%`;
      overallCircleProgress.setAttribute('stroke-dasharray', `${pct}, 100`);

      if (pct >= 70) {
        overallCircleProgress.style.stroke = 'var(--color-normal)';
        healthScoreBadge.className = 'health-badge-high';
        healthScoreBadge.innerText = `Tốt · ${pct}%`;
      } else if (pct >= 40) {
        overallCircleProgress.style.stroke = 'var(--color-warning)';
        healthScoreBadge.className = 'health-badge-medium';
        healthScoreBadge.innerText = `Suy giảm · ${pct}%`;
      } else {
        overallCircleProgress.style.stroke = 'var(--color-artifact)';
        healthScoreBadge.className = 'health-badge-low';
        healthScoreBadge.innerText = `Kém · ${pct}%`;
      }

      // Cập nhật 5 thanh đo thành phần
      valBlur.innerText = `${(ch.blur_health * 100).toFixed(0)}%`;
      barBlur.style.width = `${Math.round(ch.blur_health * 100)}%`;
      barBlur.className = `progress-fill ${getBarClass(ch.blur_health)}`;

      valBright.innerText = `${(ch.brightness_health * 100).toFixed(0)}%`;
      barBright.style.width = `${Math.round(ch.brightness_health * 100)}%`;
      barBright.className = `progress-fill ${getBarClass(ch.brightness_health)}`;

      valOccl.innerText = `${(ch.occlusion_health * 100).toFixed(0)}%`;
      barOccl.style.width = `${Math.round(ch.occlusion_health * 100)}%`;
      barOccl.className = `progress-fill ${getBarClass(ch.occlusion_health)}`;

      valNoise.innerText = `${(ch.noise_health * 100).toFixed(0)}%`;
      barNoise.style.width = `${Math.round(ch.noise_health * 100)}%`;
      barNoise.className = `progress-fill ${getBarClass(ch.noise_health)}`;

      valFreeze.innerText = `${(ch.freeze_health * 100).toFixed(0)}%`;
      barFreeze.style.width = `${Math.round(ch.freeze_health * 100)}%`;
      barFreeze.className = `progress-fill ${getBarClass(ch.freeze_health)}`;

      // Cập nhật nhãn chuẩn đoán
      if (ch.status_tags && ch.status_tags.length > 0) {
        healthTagsContainer.innerHTML = ch.status_tags.map(t => {
          const translatedTag = (TAG_TRANSLATIONS[t] || t).replace(/_/g, ' ').toLocaleLowerCase('vi');
          const vnTag = translatedTag.charAt(0).toLocaleUpperCase('vi') + translatedTag.slice(1);
          let cls = 'tag-healthy';
          if (t.includes('BLUR') || t.includes('DARK') || t.includes('SMUDGE') || t.includes('NOISE') || t.includes('LOW_LIGHT')) {
            cls = 'tag-warning';
          }
          if (t.includes('FROZEN') || t.includes('HEAVY') || t.includes('OCCLUSION')) {
            cls = 'tag-critical';
          }
          return `<span class="diag-tag ${cls}">${vnTag}</span>`;
        }).join('');
      }
    }

    // Cập nhật Bảng Theo Dõi Đối Tượng ByteTrack
    if (data.tracks) {
      activeTrackCount.innerText = `${data.tracks.length} đối tượng`;
      if (data.tracks.length === 0) {
        tracksTableBody.innerHTML = '<tr class="empty-row"><td colspan="7">Không phát hiện người đi bộ trong khung hình.</td></tr>';
      } else {
        const predMap = {};
        if (data.predictions) {
          data.predictions.forEach(p => { predMap[p.track_id] = p; });
        }

        tracksTableBody.innerHTML = data.tracks.map(t => {
          const p = predMap[t.track_id];
          const rawPred = p ? p.prediction : (t.is_stationary ? 'CAMERA_ARTIFACT' : 'NORMAL');
          const stateLabel = DECISION_LABELS[rawPred] || rawPred;

          let stateColor = 'var(--text-muted)';
          if (rawPred === 'NORMAL') stateColor = 'var(--color-normal)';
          if (rawPred === 'REAL_RARE_EVENT') stateColor = 'var(--color-rare)';
          if (rawPred === 'CAMERA_ARTIFACT') stateColor = 'var(--color-artifact)';

          const motionDisplay = t.is_stationary 
            ? `<span style="color: var(--color-artifact); font-weight: 500;">Đứng yên (${t.motion_score.toFixed(2)})</span>` 
            : `${t.motion_score.toFixed(2)} px`;

          return `
            <tr>
              <td><strong style="color: var(--accent-blue)">#${t.track_id}</strong></td>
              <td>[${t.bbox.join(', ')}]</td>
              <td>${(t.confidence * 100).toFixed(0)}%</td>
              <td>${(t.tracking_consistency * 100).toFixed(0)}%</td>
              <td>${motionDisplay}</td>
              <td>${t.persistence}</td>
              <td style="color: ${stateColor}; font-weight: 500;">${stateLabel}</td>
            </tr>
          `;
        }).join('');
      }
    }

    // Cập nhật xác suất phân loại
    if (data.predictions && data.predictions.length > 0) {
      const p = data.predictions[0];
      const probs = p.probabilities || {};

      const pNorm = Math.round((probs.NORMAL || 0) * 100);
      const pRare = Math.round((probs.REAL_RARE_EVENT || 0) * 100);
      const pArt = Math.round((probs.CAMERA_ARTIFACT || 0) * 100);

      probNormalVal.innerText = `${pNorm}%`;
      probNormalFill.style.width = `${pNorm}%`;

      probRareVal.innerText = `${pRare}%`;
      probRareFill.style.width = `${pRare}%`;

      probArtifactVal.innerText = `${pArt}%`;
      probArtifactFill.style.width = `${pArt}%`;
    } else {
      probNormalVal.innerText = "0%";
      probNormalFill.style.width = "0%";
      probRareVal.innerText = "0%";
      probRareFill.style.width = "0%";
      probArtifactVal.innerText = "0%";
      probArtifactFill.style.width = "0%";
    }

    // Cập nhật Nhật Ký Sự Kiện
    if (data.events && data.events.length > 0) {
      logFeed.innerHTML = data.events.slice(0, 15).map(ev => {
        let cls = 'log-info';
        if (ev.level === 'warning') cls = 'log-warning';
        if (ev.level === 'danger') cls = 'log-danger';
        if (ev.level === 'success') cls = 'log-success';

        let message = ev.message;
        if (ev.type === 'SYSTEM') message = 'Hệ thống đã khởi động';
        if (ev.type === 'SCENARIO' || ev.type === 'SCENARIO_SWITCH') {
          const scenarioKeys = Object.keys(SCENARIO_LABELS);
          const scenarioNumber = ev.message.match(/Tình huống (\d)/);
          const scenario = scenarioKeys.find(key => ev.message.includes(key)) || (scenarioNumber && scenarioKeys[Number(scenarioNumber[1]) - 1]);
          message = scenario ? `Tình huống: ${SCENARIO_LABELS[scenario]}` : ev.message.replace(/^(Đã chuyển sang:|Đang phát:)\s*/, 'Tình huống: ');
        }
        if (ev.type === 'DECISION_ALERT') {
          const decision = Object.keys(DECISION_LABELS).find(key => ev.message.includes(key))
            || (ev.message.includes('BÌNH THƯỜNG') ? 'NORMAL'
              : ev.message.includes('HIẾM CÓ THẬT') ? 'REAL_RARE_EVENT'
              : ev.message.includes('BÁO ĐỘNG GIẢ') ? 'CAMERA_ARTIFACT' : null);
          if (decision) message = DECISION_LABELS[decision];
        }

        return `
          <div class="log-item ${cls}">
            <span class="log-time">${ev.timestamp}</span>
            <span class="log-msg">${message}</span>
          </div>
        `;
      }).join('');
    }
  }

  // Quét danh sách video đã tải lên và khởi tạo polling
  fetchUploadedVideosList();
  pollInterval = setInterval(fetchTelemetry, 200);
  fetchTelemetry();
});
