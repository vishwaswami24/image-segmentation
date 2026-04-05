const uploadInput = document.getElementById("imageFile");
const uploadLabel = document.getElementById("uploadLabel");
const clearButton = document.getElementById("clearButton");

if (clearButton) {
  clearButton.addEventListener("click", () => {
    window.location.href = "/";
  });
}

if (uploadInput && uploadLabel) {
  uploadInput.addEventListener("change", () => {
    const file = uploadInput.files && uploadInput.files[0];
    uploadLabel.textContent = file ? file.name : "Drop an image here or click to browse";
  });
}

document.querySelectorAll(".range-input").forEach((input) => {
  const output = input.parentElement.querySelector("output");
  if (!output) {
    return;
  }

  const sync = () => {
    const suffix = output.id === "coverageFilterOutput" ? "%" : "";
    output.textContent = `${input.value}${suffix}`;
  };

  sync();
  input.addEventListener("input", sync);
});

document.querySelectorAll(".preset-chip").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll(".preset-chip").forEach((chip) => {
      chip.classList.toggle("is-active", chip === button);
    });

    const scoreInput = document.querySelector('input[name="score_threshold"]');
    const maskInput = document.querySelector('input[name="mask_threshold"]');
    const topKInput = document.querySelector('input[name="top_k"]');

    if (scoreInput && button.dataset.presetScore) {
      scoreInput.value = button.dataset.presetScore;
      scoreInput.dispatchEvent(new Event("input", { bubbles: true }));
    }

    if (maskInput && button.dataset.presetMask) {
      maskInput.value = button.dataset.presetMask;
      maskInput.dispatchEvent(new Event("input", { bubbles: true }));
    }

    if (topKInput && button.dataset.presetTopk) {
      topKInput.value = button.dataset.presetTopk;
      topKInput.dispatchEvent(new Event("input", { bubbles: true }));
    }
  });
});

const resultDataNode = document.getElementById("resultData");

if (resultDataNode) {
  const result = JSON.parse(resultDataNode.textContent);
  const maskGrid = document.getElementById("maskGrid");
  const filteredEmpty = document.getElementById("filteredEmpty");
  const searchInput = document.getElementById("searchInput");
  const labelFilter = document.getElementById("labelFilter");
  const sortSelect = document.getElementById("sortSelect");
  const coverageFilter = document.getElementById("coverageFilter");
  const labelChips = document.querySelectorAll(".label-chip");

  const focusLabel = document.getElementById("focusLabel");
  const focusSwatch = document.getElementById("focusSwatch");
  const focusMaskImage = document.getElementById("focusMaskImage");
  const focusScore = document.getElementById("focusScore");
  const focusCoverage = document.getElementById("focusCoverage");
  const focusPixels = document.getElementById("focusPixels");
  const focusBox = document.getElementById("focusBox");
  const focusDownload = document.getElementById("focusDownload");

  const cards = Array.from(document.querySelectorAll(".interactive-mask"));
  let activeIndex = cards.findIndex((card) => card.classList.contains("is-active"));
  if (activeIndex < 0) {
    activeIndex = 0;
  }

  const setFocus = (card) => {
    if (!card) {
      return;
    }

    cards.forEach((item) => item.classList.toggle("is-active", item === card));
    activeIndex = cards.indexOf(card);

    if (!focusLabel) {
      return;
    }

    focusLabel.textContent = card.dataset.label;
    focusSwatch.style.background = card.dataset.color || "transparent";
    focusMaskImage.src = card.dataset.maskUrl;
    focusMaskImage.alt = `Focused segmentation mask for ${card.dataset.label}`;
    focusScore.textContent = `${Number(card.dataset.score).toFixed(2)}%`;
    focusCoverage.textContent = `${Number(card.dataset.coverage).toFixed(2)}%`;
    focusPixels.textContent = card.dataset.maskPixels;
    focusBox.textContent = `Box: ${card.dataset.box}`;
    focusDownload.href = card.dataset.maskUrl;
  };

  const applyFilters = () => {
    if (!cards.length || !maskGrid || !searchInput || !sortSelect || !coverageFilter) {
      if (filteredEmpty) {
        filteredEmpty.hidden = true;
      }
      return;
    }

    const activeChip = document.querySelector(".label-chip.is-active");
    const chipLabel = activeChip ? activeChip.dataset.chipLabel : "all";
    const selectedLabel =
      chipLabel && chipLabel !== "all"
        ? chipLabel
        : (labelFilter ? labelFilter.value : "all");
    const searchTerm = searchInput.value.trim().toLowerCase();
    const coverageThreshold = Number(coverageFilter.value || 0);

    const visibleCards = cards.filter((card) => {
      const label = (card.dataset.label || "").toLowerCase();
      const matchesSearch = !searchTerm || label.includes(searchTerm);
      const matchesLabel =
        selectedLabel === "all" || !selectedLabel || card.dataset.label === selectedLabel;
      const matchesCoverage = Number(card.dataset.coverage || 0) >= coverageThreshold;
      const visible = matchesSearch && matchesLabel && matchesCoverage;
      card.classList.toggle("is-hidden", !visible);
      return visible;
    });

    const sorter = sortSelect.value;
    const sortedCards = [...visibleCards].sort((left, right) => {
      const leftScore = Number(left.dataset.score || 0);
      const rightScore = Number(right.dataset.score || 0);
      const leftCoverage = Number(left.dataset.coverage || 0);
      const rightCoverage = Number(right.dataset.coverage || 0);
      const leftLabel = left.dataset.label || "";
      const rightLabel = right.dataset.label || "";

      if (sorter === "score_asc") {
        return leftScore - rightScore;
      }
      if (sorter === "coverage_desc") {
        return rightCoverage - leftCoverage;
      }
      if (sorter === "label_asc") {
        return leftLabel.localeCompare(rightLabel);
      }
      return rightScore - leftScore;
    });

    sortedCards.forEach((card) => maskGrid.appendChild(card));

    const activeCardVisible = visibleCards.includes(cards[activeIndex]);
    if (!activeCardVisible && visibleCards[0]) {
      setFocus(visibleCards[0]);
    }

    if (filteredEmpty) {
      filteredEmpty.hidden = visibleCards.length !== 0;
    }
  };

  cards.forEach((card) => {
    card.addEventListener("click", () => setFocus(card));
    card.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        setFocus(card);
      }
    });
  });

  labelChips.forEach((chip) => {
    chip.addEventListener("click", () => {
      labelChips.forEach((item) => item.classList.remove("is-active"));
      chip.classList.add("is-active");
      if (labelFilter) {
        labelFilter.value = chip.dataset.chipLabel === "all" ? "all" : chip.dataset.chipLabel;
      }
      applyFilters();
    });
  });

  [searchInput, labelFilter, sortSelect, coverageFilter].forEach((element) => {
    if (!element) {
      return;
    }

    element.addEventListener("input", () => {
      if (element === labelFilter) {
        labelChips.forEach((chip) => {
          const isMatch =
            (chip.dataset.chipLabel || "all") === (labelFilter.value || "all");
          chip.classList.toggle("is-active", isMatch);
        });
      }
      if (element === labelFilter && labelFilter.value === "all") {
        const allChip = document.querySelector('[data-chip-label="all"]');
        if (allChip) {
          allChip.classList.add("is-active");
        }
      }
      applyFilters();
    });
  });

  const comparisonOverlay = document.getElementById("comparisonOverlay");
  const comparisonHandle = document.getElementById("comparisonHandle");
  const comparisonStage = document.getElementById("comparisonStage");
  const comparisonSlider = document.getElementById("comparisonSlider");
  const comparisonOutput = document.getElementById("comparisonOutput");
  const viewModes = Array.from(document.querySelectorAll(".view-mode"));

  const renderComparison = (mode, sliderValue) => {
    if (!comparisonOverlay || !comparisonHandle || !comparisonOutput || !comparisonStage) {
      return;
    }

    comparisonStage.dataset.viewMode = mode;

    if (mode === "original") {
      comparisonStage.style.setProperty("--split-point", "0%");
      comparisonOutput.textContent = "Source only";
      return;
    }

    if (mode === "overlay") {
      comparisonStage.style.setProperty("--split-point", "100%");
      comparisonOutput.textContent = "Mask overlay";
      return;
    }

    comparisonStage.style.setProperty("--split-point", `${sliderValue}%`);
    comparisonOutput.textContent = `${sliderValue}%`;
  };

  if (comparisonSlider) {
    let currentMode = "overlay";

    const syncModeButtons = () => {
      viewModes.forEach((button) => {
        button.classList.toggle("is-active", button.dataset.viewMode === currentMode);
      });
    };

    comparisonSlider.addEventListener("input", () => {
      currentMode = "split";
      syncModeButtons();
      renderComparison(currentMode, Number(comparisonSlider.value));
    });

    viewModes.forEach((button) => {
      button.addEventListener("click", () => {
        currentMode = button.dataset.viewMode || "split";
        syncModeButtons();
        renderComparison(currentMode, Number(comparisonSlider.value));
      });
    });

    renderComparison(currentMode, Number(comparisonSlider.value));
  }

  if (cards[activeIndex]) {
    setFocus(cards[activeIndex]);
  }
  applyFilters();

  if (!result.detections || result.detections.length === 0) {
    if (filteredEmpty) {
      filteredEmpty.hidden = false;
    }
  }
}
