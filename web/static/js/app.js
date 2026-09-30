/**
 * Hisse Senedi Fiyat Tahmini (PyTorch LSTM ve GRU)
 * Sade & Modern Açık Tema İstemci Mantığı
 */

let mainChart = null;
let rawChartData = null;

document.addEventListener("DOMContentLoaded", () => {
    initApp();
});

async function initApp() {
    setupTabNavigation();
    setupPredictionPlayground();
    setupTimeFilterButtons();
    
    await loadOverviewStats();
    await loadMetricsTable();
    await loadMainChart();
    await runInferenceDemo();
}

/**
 * Üst KPI kartlarını doldur
 */
async function loadOverviewStats() {
    try {
        const res = await fetch("/api/overview");
        const data = await res.json();
        
        document.getElementById("kpi-last-price").textContent = `$${data.lastPrice.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
        document.getElementById("kpi-growth").textContent = data.overallGrowth;
        document.getElementById("kpi-device").textContent = `${data.device} GPU AKTİF`;
        document.getElementById("kpi-param-reduction").textContent = `-%${parseFloat(data.paramReduction).toFixed(1)}`;
        document.getElementById("badge-device-indicator").textContent = `${data.device} HIZLANDIRICI AKTİF`;
        
        document.getElementById("overview-dates").textContent = `${data.startDate} ile ${data.endDate} arası (${data.totalDays.toLocaleString('tr-TR')} işlem günü)`;
    } catch (err) {
        console.error("Özet istatistik yükleme hatası:", err);
    }
}

/**
 * Model kıyaslama tablosunu doldur
 */
async function loadMetricsTable() {
    try {
        const res = await fetch("/api/metrics");
        const data = await res.json();
        const tbody = document.getElementById("metrics-table-body");
        tbody.innerHTML = "";
        
        data.forEach(row => {
            const tr = document.createElement("tr");
            const isWinner = row.Model === "GRU";
            
            tr.innerHTML = `
                <td class="model-cell">
                    <span class="model-badge badge-${row.Model.toLowerCase()}"></span>
                    <strong>${row.Model} Modeli</strong>
                    ${isWinner ? '<span class="kpi-badge badge-green" style="margin-left: 6px;">EN İYİ</span>' : ''}
                </td>
                <td><span style="font-family: var(--font-mono); font-weight: 600;">${row.Parameters.toLocaleString('tr-TR')}</span></td>
                <td><span class="${isWinner ? 'highlight-metric' : ''}">${row["Training Time (s)"]} sn</span></td>
                <td>$${row["Train RMSE ($)"]}</td>
                <td><span class="${isWinner ? 'highlight-metric' : ''}">$${row["Test RMSE ($)"]}</span></td>
                <td>$${row["Test MAE ($)"]}</td>
                <td><span class="${isWinner ? 'highlight-metric' : ''}">%${row["Test MAPE (%)"]}</span></td>
                <td><span class="${isWinner ? 'highlight-metric' : ''}">${row["Test R2"]}</span></td>
            `;
            tbody.appendChild(tr);
        });
    } catch (err) {
        console.error("Metrik tablosu yükleme hatası:", err);
    }
}

/**
 * Zaman serisi verilerini çekip sade ve ferah açık tema grafiğini çiz
 */
async function loadMainChart(step = 1) {
    try {
        const res = await fetch(`/api/chart-data?step=${step}`);
        rawChartData = await res.json();
        
        const ctx = document.getElementById("mainStockChart").getContext("2d");
        
        // Açık temaya uygun yumuşak mavi dolgu
        const gradientActual = ctx.createLinearGradient(0, 0, 0, 400);
        gradientActual.addColorStop(0, "rgba(37, 99, 235, 0.12)");
        gradientActual.addColorStop(1, "rgba(37, 99, 235, 0.0)");
        
        if (mainChart) {
            mainChart.destroy();
        }
        
        mainChart = new Chart(ctx, {
            type: "line",
            data: {
                labels: rawChartData.dates,
                datasets: [
                    {
                        label: "Gerçek Kapanış Fiyatı",
                        data: rawChartData.actual,
                        borderColor: "#2563eb",
                        backgroundColor: gradientActual,
                        borderWidth: 2,
                        fill: true,
                        pointRadius: 0,
                        pointHoverRadius: 5,
                        tension: 0.1
                    },
                    {
                        label: "GRU Test Tahmini (En İyi)",
                        data: rawChartData.gruTest,
                        borderColor: "#4f46e5",
                        borderWidth: 2.2,
                        pointRadius: 0,
                        pointHoverRadius: 6,
                        tension: 0.1,
                        borderDash: [0, 0]
                    },
                    {
                        label: "LSTM Test Tahmini",
                        data: rawChartData.lstmTest,
                        borderColor: "#dc2626",
                        borderWidth: 1.8,
                        pointRadius: 0,
                        pointHoverRadius: 6,
                        tension: 0.1,
                        borderDash: [4, 4]
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    mode: "index",
                    intersect: false
                },
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        backgroundColor: "#0f172a",
                        titleColor: "#ffffff",
                        bodyColor: "#cbd5e1",
                        borderColor: "#e2e8f0",
                        borderWidth: 1,
                        padding: 10,
                        callbacks: {
                            label: function(context) {
                                let label = context.dataset.label || '';
                                if (label) {
                                    label += ': ';
                                }
                                if (context.parsed.y !== null) {
                                    label += '$' + context.parsed.y.toLocaleString('en-US', { minimumFractionDigits: 2 });
                                }
                                return label;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        grid: {
                            color: "rgba(0, 0, 0, 0.05)"
                        },
                        ticks: {
                            color: "#64748b",
                            maxTicksLimit: 10,
                            font: {
                                family: "'JetBrains Mono', monospace",
                                size: 11
                            }
                        }
                    },
                    y: {
                        grid: {
                            color: "rgba(0, 0, 0, 0.05)"
                        },
                        ticks: {
                            color: "#64748b",
                            callback: function(value) {
                                return '$' + value;
                            },
                            font: {
                                family: "'JetBrains Mono', monospace",
                                size: 11
                            }
                        }
                    }
                }
            }
        });
        
        renderCustomLegend();
        
    } catch (err) {
        console.error("Grafik yükleme hatası:", err);
    }
}

function renderCustomLegend() {
    const container = document.getElementById("chart-legend-container");
    container.innerHTML = "";
    
    mainChart.data.datasets.forEach((dataset, index) => {
        const item = document.createElement("div");
        item.className = "legend-item";
        item.id = `legend-item-${index}`;
        item.innerHTML = `
            <span class="legend-dot" style="background-color: ${dataset.borderColor};"></span>
            <span>${dataset.label}</span>
        `;
        
        item.addEventListener("click", () => {
            const isVisible = mainChart.isDatasetVisible(index);
            mainChart.setDatasetVisibility(index, !isVisible);
            mainChart.update();
            item.style.opacity = isVisible ? "0.35" : "1.0";
        });
        
        container.appendChild(item);
    });
}

/**
 * Zaman aralığı butonları (1Y, 3Y, 5Y, Tümü)
 */
function setupTimeFilterButtons() {
    const buttons = document.querySelectorAll(".time-btn");
    buttons.forEach(btn => {
        btn.addEventListener("click", () => {
            buttons.forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            
            const range = btn.getAttribute("data-range");
            filterChartRange(range);
        });
    });
}

function filterChartRange(range) {
    if (!rawChartData || !mainChart) return;
    
    let totalPoints = rawChartData.dates.length;
    let sliceStart = 0;
    
    if (range === "1Y") sliceStart = Math.max(0, totalPoints - 252);
    else if (range === "3Y") sliceStart = Math.max(0, totalPoints - 252 * 3);
    else if (range === "5Y") sliceStart = Math.max(0, totalPoints - 252 * 5);
    else sliceStart = 0;
    
    mainChart.data.labels = rawChartData.dates.slice(sliceStart);
    mainChart.data.datasets[0].data = rawChartData.actual.slice(sliceStart);
    mainChart.data.datasets[1].data = rawChartData.gruTest.slice(sliceStart);
    mainChart.data.datasets[2].data = rawChartData.lstmTest.slice(sliceStart);
    
    mainChart.update();
}

/**
 * İnteraktif Çıkarım (Playground)
 */
function setupPredictionPlayground() {
    const btn = document.getElementById("btn-run-inference");
    const slider = document.getElementById("price-shock-slider");
    const sliderVal = document.getElementById("slider-val-display");
    
    if (slider) {
        slider.addEventListener("input", (e) => {
            const val = parseFloat(e.target.value);
            sliderVal.textContent = (val > 0 ? `+%${val}` : `%${val}`);
        });
    }
    
    if (btn) {
        btn.addEventListener("click", async () => {
            await runInferenceDemo();
        });
    }
}

async function runInferenceDemo() {
    const btn = document.getElementById("btn-run-inference");
    const slider = document.getElementById("price-shock-slider");
    const shockPct = slider ? parseFloat(slider.value) : 0;
    
    btn.innerHTML = `Model Hesaplanıyor...`;
    btn.disabled = true;
    
    try {
        let payload = {};
        if (shockPct !== 0 && rawChartData) {
            const recent = rawChartData.actual.slice(-20);
            const shocked = recent.map(p => p * (1 + shockPct / 100));
            payload = { sequence: shocked };
        }
        
        const res = await fetch("/api/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });
        
        const data = await res.json();
        
        document.getElementById("pred-current-price").textContent = `$${data.currentPrice.toLocaleString('en-US', { minimumFractionDigits: 2 })}`;
        
        // GRU (Kazanan)
        document.getElementById("pred-gru-price").textContent = `$${data.gru.predictedPrice.toLocaleString('en-US', { minimumFractionDigits: 2 })}`;
        const gruChangeEl = document.getElementById("pred-gru-change");
        gruChangeEl.textContent = `${data.gru.change >= 0 ? '+' : ''}${data.gru.change} USD (${data.gru.pctChange >= 0 ? '+' : ''}%${data.gru.pctChange})`;
        gruChangeEl.className = `pred-change ${data.gru.change >= 0 ? 'change-up' : 'change-down'}`;
        
        // LSTM
        document.getElementById("pred-lstm-price").textContent = `$${data.lstm.predictedPrice.toLocaleString('en-US', { minimumFractionDigits: 2 })}`;
        const lstmChangeEl = document.getElementById("pred-lstm-change");
        lstmChangeEl.textContent = `${data.lstm.change >= 0 ? '+' : ''}${data.lstm.change} USD (${data.lstm.pctChange >= 0 ? '+' : ''}%${data.lstm.pctChange})`;
        lstmChangeEl.className = `pred-change ${data.lstm.change >= 0 ? 'change-up' : 'change-down'}`;
        
        // Gecikme
        document.getElementById("inference-latency").textContent = `${data.latencyMs} ms (${data.device})`;
        
    } catch (err) {
        console.error("Çıkarım hatası:", err);
    } finally {
        btn.innerHTML = `<span>Model Tahminini Çalıştır</span>`;
        btn.disabled = false;
    }
}

/**
 * Mimari Sekme Geçişi
 */
function setupTabNavigation() {
    const tabs = document.querySelectorAll(".tab-btn");
    tabs.forEach(tab => {
        tab.addEventListener("click", () => {
            tabs.forEach(t => t.classList.remove("active"));
            tab.classList.add("active");
            
            const targetId = tab.getAttribute("data-target");
            document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));
            const targetContent = document.getElementById(targetId);
            if (targetContent) {
                targetContent.classList.add("active");
            }
        });
    });
}
