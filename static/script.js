
"use strict";

/* ============================================
   STOCKWISE — INVENTORY INTELLIGENCE
   Data source: Flask API
   No demo data or hardcoded dashboard values
   ============================================ */

let trendChart = null;
let statusChart = null;

const uploadForm = document.getElementById("uploadForm");
const uploadMessage = document.getElementById("uploadMessage");
const csvFile = document.getElementById("csvFile");
const exportButton = document.getElementById("exportButton");

const numberFormat = new Intl.NumberFormat("en-IN", {
    maximumFractionDigits: 2
});

/* --------------------------------------------
   Utility functions
-------------------------------------------- */

function formatNumber(value) {
    if (value === null || value === undefined || value === "") {
        return "—";
    }

    const number = Number(value);

    return Number.isFinite(number)
        ? numberFormat.format(number)
        : "—";
}

function displayValue(value) {
    if (value === null || value === undefined || value === "") {
        return "—";
    }

    return String(value);
}

function showMessage(message, type = "") {
    if (!uploadMessage) return;

    uploadMessage.textContent = message;
    uploadMessage.className = type;
}

async function readJsonResponse(response) {
    const body = await response.text();
    let data = {};

    try {
        data = body ? JSON.parse(body) : {};
    } catch (error) {
        if (response.ok) {
            throw new Error("The server returned an invalid response.");
        }
    }

    if (!response.ok) {
        const status = `${response.status} ${response.statusText}`.trim();
        const message = response.status === 413
            ? "The upload exceeded Vercel's 4.5 MB request limit."
            : `The server returned HTTP ${status}.`;

        throw new Error(data.error || message);
    }

    if (!body) {
        throw new Error("The server returned an empty response.");
    }

    return data;
}

async function gzipCsvFile(file) {
    if (!window.CompressionStream) {
        throw new Error("This browser cannot compress CSV uploads. Please use a current browser.");
    }

    const compressedStream = file.stream().pipeThrough(
        new CompressionStream("gzip")
    );
    const compressedBlob = await new Response(compressedStream).blob();

    return new File([compressedBlob], file.name, {
        type: "application/gzip"
    });
}

function destroyChart(chart) {
    if (chart) {
        chart.destroy();
    }
}

/* --------------------------------------------
   Clear dashboard before data is uploaded
-------------------------------------------- */

function clearDashboard() {
    document.getElementById("totalProducts").textContent = "—";
    document.getElementById("atRisk").textContent = "—";
    document.getElementById("critical").textContent = "—";
    document.getElementById("totalInventory").textContent = "—";

    document.getElementById("recordCount").textContent = "0 records";

    document.getElementById("inventoryBody").innerHTML = `
        <tr>
            <td colspan="8" class="table-empty">
                Upload a CSV to analyze inventory records.
            </td>
        </tr>
    `;

    document.getElementById("statusLegend").replaceChildren();

    const trendCanvas = document.getElementById("trendChart");
    const trendEmpty = document.getElementById("trendEmpty");
    const statusCanvas = document.getElementById("statusChart");

    destroyChart(trendChart);
    destroyChart(statusChart);

    trendChart = null;
    statusChart = null;

    trendCanvas.style.display = "none";
    trendEmpty.style.display = "block";
    trendEmpty.textContent = "Upload a dated CSV to view sales trends.";

    statusCanvas.style.display = "none";

    // Clear any previous message only when explicitly requested
    // by the dashboard initialization.
}

/* --------------------------------------------
   Update KPI cards from Flask metrics
-------------------------------------------- */

function updateMetrics(metrics) {
    if (!metrics) {
        clearDashboard();
        return;
    }

    // Support the property names used by the Flask backend.
    const totalRecords =
        metrics.total_records ??
        metrics.total_products ??
        0;

    const atRisk =
        metrics.at_risk ??
        0;

    const critical =
        metrics.critical ??
        0;

    const totalInventory =
        metrics.total_inventory ??
        0;

    document.getElementById("totalProducts").textContent =
        formatNumber(totalRecords);

    document.getElementById("atRisk").textContent =
        formatNumber(atRisk);

    document.getElementById("critical").textContent =
        formatNumber(critical);

    document.getElementById("totalInventory").textContent =
        formatNumber(totalInventory);
}

/* --------------------------------------------
   Sales trend chart
   Expected API format:
   [{ "date": "2026-01-01", "units": 25 }]
-------------------------------------------- */

function renderTrend(trend) {
    const canvas = document.getElementById("trendChart");
    const empty = document.getElementById("trendEmpty");

    destroyChart(trendChart);
    trendChart = null;

    if (!Array.isArray(trend) || trend.length === 0) {
        canvas.style.display = "none";
        empty.style.display = "block";
        empty.textContent = "No dated sales data available.";
        return;
    }

    const validTrend = trend.filter(item =>
        item.date !== null &&
        item.date !== undefined &&
        item.units !== null &&
        item.units !== undefined &&
        Number.isFinite(Number(item.units))
    );

    if (validTrend.length === 0) {
        canvas.style.display = "none";
        empty.style.display = "block";
        empty.textContent = "No valid sales trend data available.";
        return;
    }

    canvas.style.display = "block";
    empty.style.display = "none";

    trendChart = new Chart(canvas, {
        type: "line",

        data: {
            labels: validTrend.map(item => item.date),

            datasets: [{
                label: "Units sold",
                data: validTrend.map(item => Number(item.units)),

                borderColor: "#702D3B",
                backgroundColor: "rgba(112, 45, 59, 0.10)",

                fill: true,
                tension: 0.35,

                pointRadius: 3,
                pointHoverRadius: 6,
                pointBackgroundColor: "#702D3B",
                pointBorderColor: "#FFF9F3",

                borderWidth: 2
            }]
        },

        options: {
            responsive: true,
            maintainAspectRatio: false,

            interaction: {
                intersect: false,
                mode: "index"
            },

            animation: {
                duration: 700,
                easing: "easeOutQuart"
            },

            plugins: {
                legend: {
                    display: false
                },

                tooltip: {
                    backgroundColor: "#4B1C2A",
                    titleColor: "#FFF5EC",
                    bodyColor: "#F3E9DE",
                    padding: 12,
                    cornerRadius: 8
                }
            },

            scales: {
                x: {
                    grid: {
                        display: false
                    },

                    ticks: {
                        maxTicksLimit: 7,
                        color: "#958078"
                    },

                    border: {
                        display: false
                    }
                },

                y: {
                    beginAtZero: true,

                    ticks: {
                        precision: 0,
                        color: "#958078"
                    },

                    grid: {
                        color: "rgba(112, 45, 59, 0.08)"
                    },

                    border: {
                        display: false
                    }
                }
            }
        }
    });
}

/* --------------------------------------------
   Stock status doughnut chart
-------------------------------------------- */

function renderStatusChart(counts) {
    const canvas = document.getElementById("statusChart");
    const legend = document.getElementById("statusLegend");

    destroyChart(statusChart);
    statusChart = null;

    legend.replaceChildren();

    const labels = ["Low Risk", "At Risk"];
    const colors = ["#648D72", "#A7545C"];

    const values = labels.map(label => {
        const value = Number(counts?.[label] ?? 0);
        return Number.isFinite(value) && value > 0 ? value : 0;
    });

    const total = values.reduce((sum, value) => sum + value, 0);

    if (total === 0) {
        canvas.style.display = "none";

        const message = document.createElement("p");
        message.textContent = "No stock status data available yet.";
        legend.appendChild(message);

        return;
    }

    canvas.style.display = "block";

    statusChart = new Chart(canvas, {
        type: "doughnut",

        data: {
            labels,

            datasets: [{
                data: values,
                backgroundColor: colors,
                borderColor: "#FFF9F3",
                borderWidth: 3,
                hoverOffset: 7
            }]
        },

        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: "70%",

            animation: {
                animateRotate: true,
                duration: 700
            },

            plugins: {
                legend: {
                    display: false
                },

                tooltip: {
                    backgroundColor: "#4B1C2A",
                    titleColor: "#FFF5EC",
                    bodyColor: "#F3E9DE",
                    padding: 12,
                    cornerRadius: 8,

                    callbacks: {
                        label(context) {
                            return `${context.label}: ${formatNumber(context.raw)}`;
                        }
                    }
                }
            }
        }
    });

    labels.forEach((label, index) => {
        const row = document.createElement("div");
        row.className = "legend-item";

        const left = document.createElement("span");
        left.style.display = "inline-flex";
        left.style.alignItems = "center";
        left.style.gap = "7px";

        const dot = document.createElement("span");
        dot.style.display = "inline-block";
        dot.style.width = "9px";
        dot.style.height = "9px";
        dot.style.borderRadius = "50%";
        dot.style.backgroundColor = colors[index];

        const labelText = document.createElement("span");
        labelText.textContent = label;

        const value = document.createElement("strong");
        value.textContent = formatNumber(values[index]);

        left.append(dot, labelText);
        row.append(left, value);
        legend.appendChild(row);
    });
}

/* --------------------------------------------
   Inventory table
   Uses the real records returned by Flask
-------------------------------------------- */

function renderInventory(rows) {
    const tbody = document.getElementById("inventoryBody");
    const recordCount = document.getElementById("recordCount");

    tbody.replaceChildren();

    const inventoryRows = Array.isArray(rows) ? rows : [];

    recordCount.textContent =
        `${inventoryRows.length} records`;

    if (inventoryRows.length === 0) {
        const tr = document.createElement("tr");
        const td = document.createElement("td");

        td.colSpan = 8;
        td.className = "table-empty";
        td.textContent = "No inventory records available.";

        tr.appendChild(td);
        tbody.appendChild(tr);
        return;
    }

    inventoryRows.forEach(row => {
        const tr = document.createElement("tr");

        const values = [
            row["Date"],
            row["Store ID"],
            row["Product ID"],
            row["Inventory Level"],
            row["Units Sold"],
            row["Demand Forecast"],
            row["Estimated Coverage"]
        ];

        values.forEach(value => {
            const td = document.createElement("td");
            td.textContent = displayValue(value);
            tr.appendChild(td);
        });

        const statusCell = document.createElement("td");
        const badge = document.createElement("span");

        // Prefer operational stock status to match the HTML note.
        const status =
            row["Stock Status"] ??
            row["Operational Status"] ??
            row["ML Risk Status"] ??
            "Unknown";

        badge.className = "status-badge";

        const normalizedStatus = String(status).toLowerCase();

        if (
            normalizedStatus.includes("critical") ||
            normalizedStatus.includes("at risk") ||
            normalizedStatus.includes("high risk")
        ) {
            badge.classList.add("at-risk");
        }

        badge.textContent = status;

        statusCell.appendChild(badge);
        tr.appendChild(statusCell);
        tbody.appendChild(tr);
    });
}

/* --------------------------------------------
   Load dashboard from the Flask API
-------------------------------------------- */

async function loadDashboard() {
    try {
        const response = await fetch("/api/dashboard", {
            method: "GET",
            headers: {
                "Accept": "application/json"
            },
            cache: "no-store"
        });

        const data = await readJsonResponse(response);

        // Do not show old data when no file has been uploaded.
        if (!data.loaded || !data.metrics) {
            clearDashboard();
            return;
        }

        updateMetrics(data.metrics);
        renderTrend(data.trend || []);
        renderStatusChart(data.status_counts || {});
        renderInventory(data.inventory || []);

    } catch (error) {
        console.error("Dashboard loading error:", error);

        showMessage(
            `Could not load dashboard: ${error.message}`,
            "error"
        );
    }
}

/* --------------------------------------------
   Upload and analyze CSV
-------------------------------------------- */

if (uploadForm) {
    uploadForm.addEventListener("submit", async event => {
        event.preventDefault();

        if (!csvFile.files || csvFile.files.length === 0) {
            showMessage("Please select a CSV file first.", "error");
            return;
        }

        const file = csvFile.files[0];

        if (!file.name.toLowerCase().endsWith(".csv")) {
            showMessage("Please upload a CSV file.", "error");
            return;
        }

        const button = uploadForm.querySelector('button[type="submit"]');

        const originalButtonText = button
            ? button.textContent
            : "Analyze dataset";

        if (button) {
            button.disabled = true;
            button.textContent = "Analyzing...";
        }

        showMessage("Uploading and analyzing your dataset...", "loading");

        try {
            const compressedFile = await gzipCsvFile(file);
            if (compressedFile.size > 4 * 1024 * 1024) {
                throw new Error(
                    "This CSV is still too large after compression for Vercel's upload limit."
                );
            }

            const formData = new FormData();
            formData.append("file", compressedFile, file.name);

            const response = await fetch("/api/upload", {
                method: "POST",
                headers: {
                    "X-File-Compression": "gzip"
                },
                body: formData
            });

            const data = await readJsonResponse(response);

            showMessage(
                `Analysis completed. ${formatNumber(data.rows)} valid rows loaded.`,
                "success"
            );

            // Fetch the latest predictions and dashboard metrics.
            await loadDashboard();

        } catch (error) {
            console.error("Upload error:", error);

            showMessage(
                `Upload failed: ${error.message}`,
                "error"
            );

        } finally {
            if (button) {
                button.disabled = false;
                button.textContent = originalButtonText;
            }
        }
    });
}

/* --------------------------------------------
   Export analyzed CSV
-------------------------------------------- */

if (exportButton) {
    exportButton.addEventListener("click", async () => {
        exportButton.disabled = true;

        try {
            const response = await fetch("/api/export", {
                method: "GET"
            });

            if (!response.ok) {
                const data = await response.json().catch(() => ({}));
                throw new Error(
                    data.error || "Upload and analyze a CSV before exporting."
                );
            }

            const blob = await response.blob();
            const downloadUrl = URL.createObjectURL(blob);

            const link = document.createElement("a");
            link.href = downloadUrl;
            link.download = "stockwise_analysis.csv";

            document.body.appendChild(link);
            link.click();
            link.remove();

            URL.revokeObjectURL(downloadUrl);

        } catch (error) {
            showMessage(error.message, "error");

        } finally {
            exportButton.disabled = false;
        }
    });
}

/* --------------------------------------------
   Initial page load
-------------------------------------------- */

document.addEventListener("DOMContentLoaded", () => {
    clearDashboard();
    loadDashboard();
});
