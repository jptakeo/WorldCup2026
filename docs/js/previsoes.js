// Mapeia cada painel de mata-mata para o texto exibido quando ainda não há confronto definido.
const PLACEHOLDERS = {
    'panel-r32': 'da <b>16-avos</b>',
    'panel-oitavas': 'das <b>Oitavas de Final</b>',
    'panel-quartas': 'das <b>Quartas de Final</b>',
    'panel-semis': 'da <b>Semifinal</b>',
    'panel-final': 'da <b>Final e Disputa pelo 3º lugar</b>'
};

// Monta o HTML do aviso usado quando as probabilidades daquela fase ainda não estão disponíveis.
function makePlaceholder(panelId, stageLabel) {
    return `
        <div class="placeholder-box">
            <div class="placeholder-icon-wrap">
                🛠️
            </div>
            <div class="placeholder-title">
                Probabilidades de placares em breve!
            </div>
            <div class="placeholder-text">
                As probabilidades de placares ${stageLabel} serão disponibilizadas quando os confrontos forem definidos.
            </div>
        </div>
    `;
}

// Insere o placeholder no painel informado, se ele existir na página.
function renderPlaceholder(panelId, stageLabel) {
    const panel = document.getElementById(panelId);
    if (!panel) return;
    panel.innerHTML = makePlaceholder(panelId, stageLabel);
}

//const GROUPS_PROBS = 'https://raw.githubusercontent.com/BrazilianFootball/WorldCup2026/main/data/summary.csv';
const GROUPS_PROBS = 'csv/previsoes/summary.csv';
const MATCHES_CSV_URL = 'csv/previsoes/partidas.csv';
const FLAGS_CSV_URL = 'images/flags/flag.csv';
const SCORE_STAGES = [
    {panelId: 'panel-r32',     groupValue: 'R32',       showFilters: true,  gridClass: 'scorecards-grid'},
    {panelId: 'panel-oitavas', groupValue: 'oitavas',   showFilters: true,  gridClass: 'scorecards-grid'},
    {panelId: 'panel-quartas', groupValue: 'quartas',   showFilters: false, gridClass: 'scorecards-grid scorecards-grid-two'},
    {panelId: 'panel-semis',   groupValue: 'semifinal', showFilters: false, gridClass: 'scorecards-grid scorecards-grid-two'},
    {panelId: 'panel-final',   groupValue: 'final',     showFilters: false, gridClass: 'scorecards-grid scorecards-grid-two'}
];

const NUMBER_WORDS = ['zero', 'one', 'two', 'three', 'four'];

function scoreKey(h, a) {
    return `${NUMBER_WORDS[h]}_${NUMBER_WORDS[a]}`;
}

const SCORES = Array.from({ length: 25 }, (_, i) => {
    const h = Math.floor(i / 5);
    const a = i % 5;
    return {
        h,
        a,
        key: scoreKey(h, a),
        label: `${h}x${a}`
    };
});

function escapeHTML(value) {
    return String(value ?? '')
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#039;');
}

// Normaliza textos para busca: remove acentos e ignora maiúsculas/minúsculas.
function normalizeName(value) {
    return String(value ?? '')
        .trim()
        .normalize('NFD')
        .replace(/[\u0300-\u036f]/g, '')
        .toLowerCase();
}

// Converte valores percentuais do CSV em número, aceitando vírgula decimal e símbolo %.
function parseNumber(value) {
    if (value === undefined || value === null || value === '') return null;
    const n = Number(String(value).trim().replace('%', '').replace(',', '.'));
    return Number.isFinite(n) ? n : null;
}

function formatPct(value) {
    return Number(value).toFixed(1);
}

// Lê um CSV simples, detectando vírgula ou ponto e vírgula e preservando campos entre aspas.
function parseCSV(text) {
    const firstLine = text.split(/\r?\n/)[0] || '';
    const delimiter =
        (firstLine.match(/;/g) || []).length > (firstLine.match(/,/g) || []).length
            ? ';'
            : ',';

    const rows = [];
    let row = [];
    let cell = '';
    let insideQuotes = false;

    for (let i = 0; i < text.length; i++) {
        const char = text[i];
        const next = text[i + 1];

        if (char === '"' && next === '"') {
            cell += '"';
            i++;
            continue;
        }

        if (char === '"') {
            insideQuotes = !insideQuotes;
            continue;
        }

        if (char === delimiter && !insideQuotes) {
            row.push(cell.trim());
            cell = '';
            continue;
        }

        if ((char === '\n' || char === '\r') && !insideQuotes) {
            if (cell !== '' || row.length) {
                row.push(cell.trim());
                rows.push(row);
                row = [];
                cell = '';
            }
            if (char === '\r' && next === '\n') i++;
            continue;
        }

        cell += char;
    }

    if (cell !== '' || row.length) {
        row.push(cell.trim());
        rows.push(row);
    }

    if (!rows.length) return [];

    const headers = rows[0].map(h => h.trim());

    return rows.slice(1)
        .filter(r => r.some(Boolean))
        .map(r => {
            const obj = {};
            headers.forEach((h, i) => {
                obj[h] = (r[i] ?? '').trim();
            });
            return obj;
        });
}

// Carrega um arquivo CSV por fetch e devolve as linhas como objetos.
async function loadCSV(url) {
    const response = await fetch(url, { cache: 'no-store' });
    if (!response.ok) throw new Error(`Não foi possível carregar ${url}`);
    return parseCSV(await response.text());
}

function normalizeFlagUrl(url) {
    if (!url) return '';

    let value = String(url).trim();

    if (value.includes('github.com') && value.includes('/blob/')) {
        value = value
            .replace('https://github.com/', 'https://raw.githubusercontent.com/')
            .replace('/blob/', '/');
    }

    return value;
}

function getHomeCountry(row) {
    return row.home_country || row.home_team || row.home || '';
}

function getAwayCountry(row) {
    return row.away_country || row.away_team || row.away || '';
}

function getMatchGroup(row) {
    return row.group || row.stage || row.round || '';
}

function getSelectedValues(menu) {
    return menu
        ? [...menu.querySelectorAll('input[type="checkbox"]:checked')].map(input => input.value)
        : [];
}

function updateDateButton(button, selectedDates) {
    if (!button) return;
    const label = selectedDates.length
        ? `${selectedDates.length} data${selectedDates.length > 1 ? 's' : ''} selecionada${selectedDates.length > 1 ? 's' : ''}`
        : 'Todas as datas';

    button.innerHTML = `${label}<span>▾</span>`;
}

function closeOpenDropdowns(except = null) {
    document.querySelectorAll('.date-dropdown.open').forEach(dropdown => {
        if (dropdown !== except) dropdown.classList.remove('open');
    });
}

function getFilterPanel(target) {
    return target.closest('#panel-grupos, #panel-r32, #panel-oitavas, #panel-quartas, #panel-semis, #panel-final');
}

function applyUnifiedFilters(panel) {
    if (!panel) return;
    if (panel.id === 'panel-grupos') {
        applyGroupFilters(panel);
        return;
    }
    applyScoreFilters(panel);
}

function handleFilterEvent(event) {
    const target = event.target?.closest ? event.target : event.target?.parentElement;
    if (!target) return;

    if (event.type === 'click') {
        const modeButton = target.closest('.groups-mode-btn');
        if (modeButton) {
            window.PrevisoesActions?.setGroupsMode?.(modeButton.dataset.mode);
            return;
        }

        const dropdownButton = target.closest('.date-dropdown-btn');
        if (dropdownButton) {
            const dropdown = dropdownButton.closest('.date-dropdown');
            const willOpen = dropdown && !dropdown.classList.contains('open');
            closeOpenDropdowns(dropdown);
            if (dropdown) dropdown.classList.toggle('open', willOpen);
            return;
        }

        // fechar ao clicar fora (só 1 listener global, mas não duplica)
        if (!target.closest('.date-dropdown')) {
            closeOpenDropdowns();
        }

        return;
    }

    if (event.type === 'input' && target.matches('.country-filter')) {
        applyUnifiedFilters(getFilterPanel(target));
        return;
    }

    // Datas (modo "Partidas"): ao marcar/desmarcar, aplica filtros
    if (event.type === 'change' && target.matches('.date-dropdown-menu input[type="checkbox"]')) {
        applyUnifiedFilters(getFilterPanel(target));
    }
}

function installFilterDelegationOnce() {
    if (document.documentElement.dataset.previsoesFiltersBound) return;
    ['click', 'input', 'change'].forEach(type => {
        document.addEventListener(type, handleFilterEvent);
    });
    document.documentElement.dataset.previsoesFiltersBound = '1';
}

// Aplica filtros de texto, grupo e data conforme o modo atual da aba de grupos.
function applyGroupFilters(panel = document.getElementById('panel-grupos')) {
    if (!panel) return;

    const input = panel.querySelector('.group-country-filter');
    const menu = panel.querySelector('.group-dropdown-menu');
    const btn = panel.querySelector('.group-dropdown-btn');
    const noResults = panel.querySelector('.groups-no-results');
    const dateMenu = panel.querySelector('.groups-date-dropdown-menu');
    const dateBtn = panel.querySelector('.groups-date-dropdown-btn');
    const q = normalizeName(input?.value || '');

    const selectedGroups = getSelectedValues(menu);

    // label do botão
    if (btn) {
        const n = selectedGroups.length;

        let label;
        if (n === 0) {label = 'Todos os grupos';}
        else {label = `${n} grupo${n > 1 ? 's' : ''} selecionado${n > 1 ? 's' : ''}`;}
        btn.innerHTML = `${label}<span>▾</span>`;
    }

    const mode = panel.dataset.groupsMode || (panel.dataset.groupsSource === 'previsoes' ? 'stickers' : 'games');

    let visible = 0;

    if (mode === 'games') {
        if (noResults) noResults.textContent = 'Nenhum grupo encontrado com os filtros selecionados.';

        panel.querySelectorAll('#gg .group-card').forEach(card => {
            const matchGroup = !selectedGroups.length || selectedGroups.includes(card.dataset.group);
            const matchText = !q || (card.dataset.search || '').includes(q);

            const show = matchGroup && matchText;
            card.style.display = show ? '' : 'none';
            if (show) visible++;
        });
    } else {
        if (noResults) noResults.textContent = 'Nenhuma partida encontrada com os filtros selecionados.';

        const selectedDates = getSelectedValues(dateMenu);

        updateDateButton(dateBtn, selectedDates);

        panel.querySelectorAll('#groups-scorecards .match-card').forEach(card => {
            const matchGroup = !selectedGroups.length || selectedGroups.includes(card.dataset.group);
            const matchText = !q || (card.dataset.search || '').includes(q);
            const matchDate = !selectedDates.length || selectedDates.includes(card.dataset.date);
            const show = matchGroup && matchText && matchDate;
            card.style.display = show ? '' : 'none';
            if (show) visible++;
        });
    }

    if (noResults) noResults.style.display = visible ? 'none' : 'block';
}

function applyScoreFilters(panel) {
    const countryInput = panel.querySelector('.country-filter');
    const dateMenu = panel.querySelector('.date-dropdown-menu');
    const dateButton = panel.querySelector('.date-dropdown-btn');
    const noResults = panel.querySelector('.no-results');

    const countryQuery = normalizeName(countryInput?.value || '');

    const selectedDates = getSelectedValues(dateMenu);

    updateDateButton(dateButton, selectedDates);

    let visibleCount = 0;

    panel.querySelectorAll('.match-card').forEach(card => {
        const matchesCountry =
            !countryQuery ||
            card.dataset.search.includes(countryQuery);

        const matchesDate =
            !selectedDates.length ||
            selectedDates.includes(card.dataset.date);

        const shouldShow = matchesCountry && matchesDate;

        card.style.display = shouldShow ? '' : 'none';

        if (shouldShow) visibleCount++;
    });

    if (noResults) {
        noResults.style.display = visibleCount ? 'none' : 'block';
    }
}

// ════════════════════════════════════════
// GROUP STAGE
// ════════════════════════════════════════

// Bloco isolado da fase de grupos para evitar variáveis globais desnecessárias.
(function () {
    let GROUP_STAGE_ROWS = [];

    // Busca um valor em uma linha de CSV aceitando variações de nome de coluna.
    function compactHeader(value) {
        return normalizeName(value).replace(/[^a-z0-9]/g, '');
    }

    function getCSVValue(row, possibleHeaders) {
        const wanted = new Set(possibleHeaders.map(compactHeader));

        for (const key of Object.keys(row || {})) {
            if (wanted.has(compactHeader(key))) return row[key];
        }

        return '';
    }

    function getSummaryTeam(row) {
        return getCSVValue(row, [
            'team',
            'team_name',
            'country',
            'country_pt',
            'country_name',
            'name',
            'selection',
            'selecao',
            'seleção',
            'time',
            'equipe'
        ]);
    }

    function getSummary32(row) {
        return getCSVValue(row, ['round_of_32']);
    }

    function getSummaryProbabilityPercent(row) {
        const value = parseNumber(getSummary32(row));
        if (value === null) return null;

        // Aceita tanto fração (0.335) quanto percentual (33.5).
        return value <= 1 ? value * 100 : value;
    }

    function buildGroup32Map(summaryRows) {
        const map = new Map();

        summaryRows.forEach(row => {
            const team = getSummaryTeam(row);
            const pct = getSummaryProbabilityPercent(row);

            if (!team || pct === null) return;
            map.set(normalizeName(team), pct);
        });

        return map;
    }

    function getGroupFirstPlaceProbability(team, group32Map) {
        return group32Map.get(normalizeName(team)) ?? 0;
    }

    function getGroupLetters(matchRows) {
        return [...new Set(matchRows.map(r => String(r.group ?? '').trim()))]
            .filter(g => /^[A-L]$/.test(g))
            .sort();
    }

    function ensureGroupsPhaseShell(panel, includeViewToggle) {
        if (!panel || panel.querySelector('.groups-filterbar')) return;

        const modeHTML = includeViewToggle ? `
                <div class="groups-mode">
                    <button type="button" class="groups-mode-btn active" data-mode="stickers">Figurinhas</button>
                    <button type="button" class="groups-mode-btn" data-mode="charts">Gráficos</button>
                </div>
        ` : '';

        const dateHTML = includeViewToggle ? `
                <div class="date-dropdown groups-date-dropdown">
                    <button type="button" class="date-dropdown-btn groups-date-dropdown-btn">
                        Todas as datas <span>▾</span>
                    </button>
                    <div class="date-dropdown-menu groups-date-dropdown-menu"></div>
                </div>
        ` : '';

        const filterbarClasses = includeViewToggle
            ? 'filterbar groups-filterbar has-date'
            : 'filterbar groups-filterbar chances-groups-filterbar';

        const noteHTML = includeViewToggle
            ? `
                <p class="disc groups-prob-note">
                    <span class="groups-prob-note-text">% = Probabilidade estimada de cada placar. Clique para visualizar.</span>
                </p>
            `
            : `
                <div class="texto max-780 font-16" style="margin-bottom: 16px;">
                    <span class="groups-prob-note-text">Chances das seleções de passar a fase de grupos.</span>
                </div>
            `;
            
        const filterHTML = `
            ${noteHTML}
            <div class="${filterbarClasses}">
                <div class="search-wrap">
                    <span class="search-icon">🔎</span>
                    <input class="country-filter group-country-filter" type="text" placeholder="Buscar Seleção..." autocomplete="off">
                </div>

                
                ${dateHTML}

                <div class="date-dropdown group-dropdown">
                    <button type="button" class="date-dropdown-btn group-dropdown-btn">
                        Todos os grupos <span>▾</span>
                    </button>
                    <div class="date-dropdown-menu group-dropdown-menu"></div>
                </div>

                ${modeHTML}
            </div>
            <div class="no-results groups-no-results">${includeViewToggle ? 'Nenhuma partida encontrada com os filtros selecionados.' : 'Nenhum grupo encontrado com os filtros selecionados.'}</div>
        `;

        panel.insertAdjacentHTML('afterbegin', filterHTML);
    }

    // Preenche o dropdown com as letras dos grupos disponíveis no CSV.
    function populateGroupDropdown(groups) {
        const panel = document.getElementById('panel-grupos');
        if (!panel) return;

        const menu = panel.querySelector('.group-dropdown-menu');
        if (!menu) return;

        menu.innerHTML = groups.map(g => `
            <label class="date-option">
                <input type="checkbox" value="${g}">
                <span>Grupo ${g}</span>
            </label>
        `).join('');
    }

    // Converte datas no formato dd/mm ou dd/mm/aaaa para ordenação cronológica.
    function parseDropdownDate(value) {
        const [day, month, year] = String(value).trim().split('/').map(Number);

        if (!Number.isFinite(day) || !Number.isFinite(month)) {
            return Number.POSITIVE_INFINITY;
        }

        const fullYear = Number.isFinite(year)
            ? (year < 100 ? 2000 + year : year)
            : 2026;

        return new Date(fullYear, month - 1, day).getTime();
    }

    // Preenche o dropdown de datas usando apenas jogos da fase de grupos.
    function populateGroupsDateDropdown(matchRows) {
        const panel = document.getElementById('panel-grupos');
        if (!panel) return;

        const menu = panel.querySelector('.groups-date-dropdown-menu');
        if (!menu) return;

        const stageRows = matchRows.filter(r => /^[A-L]$/.test(String(r.group ?? '').trim()));
        const dates = stageRows.map(r => String(r.date ?? '').trim()).filter(Boolean);
        const uniqueDates = [...new Set(dates)].sort((a, b) => parseDropdownDate(a) - parseDropdownDate(b));

        menu.innerHTML = uniqueDates.map(d => `
            <label class="date-option">
                <input type="checkbox" value="${d}">
                <span>${d}</span>
            </label>
        `).join('');
    }

    // Conecta os eventos de busca, dropdowns e botão de modo na aba de grupos.
    function attachGroupFilters() {
        installFilterDelegationOnce();
    }

    // Cria os cards de classificação da fase de grupos para a página de Chances.
    function buildChancesGroupCards(matchRows, summaryRows) {
        const grid = document.getElementById('gg');
        if (!grid) return;

        const panel = document.getElementById('panel-grupos');
        if (!panel) return;

        panel.dataset.groupsMode = 'games';
        ensureGroupsPhaseShell(panel, false);

        grid.innerHTML = '';

        const group32Map = buildGroup32Map(summaryRows);
        const groups = getGroupLetters(matchRows);

        if (!groups.length) {
            grid.innerHTML = `
                <div class="g-card">
                    <div class="g-head">Fase de grupos<span>—</span></div>
                    <div class="g-team"><div class="g-name">Nenhum jogo de fase de grupos encontrado no CSV.</div></div>
                </div>
            `;
            return;
        }

        groups.forEach(letter => {
            const rows = matchRows.filter(r => String(r.group ?? '').trim() === letter);
            const teams = [...new Set(rows.flatMap(r => [r.home_team, r.away_team]).filter(Boolean))];

            const ranking = teams
                .map(t => ({
                    name: t,
                    pct: getGroupFirstPlaceProbability(t, group32Map)
                }))
                .sort((a, b) => b.pct - a.pct);

            const card = document.createElement('div');
            card.className = 'g-card group-card';
            card.dataset.group = letter;
            card.dataset.search = normalizeName(teams.join(' '));
            const favorite = ranking[0]?.name || '';

            card.innerHTML = `
                <div class="g-head">
                    Grupo ${letter}
                    <span>${favorite}</span>
                </div>
            `;

            ranking.forEach((t, idx) => {
                const position = idx + 1;
                const label = position === 1 ? '1º' : position === 2 ? '2º' : position === 3 ? '3º' : '4º';
                const badgeClass = position === 1 ? 'b1' : position === 2 ? 'b2' : position === 3 ? 'b3' : 'b4';
                const statusClass = position <= 2 ? 'qualify' : position === 3 ? 'playoff' : 'elim';
                const probDisplay = Math.round(t.pct * 10) / 10;
                const probText = Number.isInteger(probDisplay) ? `${probDisplay.toFixed(0)}%` : `${probDisplay}%`;

                const row = document.createElement('div');
                row.className = `g-team ${statusClass}`;
                row.innerHTML = `
                    <div class="g-row">
                        <div class="g-badge ${badgeClass}">${label}</div>
                        <div class="g-name">
                            ${t.name}
                            ${t.name === favorite ? '<span class="g-fav">🔥</span>' : ''}
                        </div>
                        <div class="g-prob">${probText}</div>
                    </div>
                    <div class="g-progress">
                        <div class="g-fill" data-width="${probDisplay}"></div>
                    </div>
                `;

                if (typeof showStats === 'function' && typeof selectTeam === 'function') {
                    row.addEventListener('mouseenter', () => showStats(t.name));
                    row.addEventListener('mouseleave', () => {
                        const back = (typeof window.selectedTeam !== 'undefined' && window.selectedTeam) ? window.selectedTeam : favorite;
                        showStats(back);
                    });
                    row.addEventListener('click', () => selectTeam(t.name));
                }

                card.appendChild(row);
            });

            grid.appendChild(card);
        });

        populateGroupDropdown(groups);
        attachGroupFilters();
        applyGroupFilters(panel);

        setTimeout(() => {
            document.querySelectorAll('#gg .g-fill').forEach(el => {
                el.style.width = el.dataset.width + '%';
            });
        }, 100);
    }

    // Renderiza os cards de partidas da fase de grupos para Previsões, separados em Figurinhas e Gráficos.
    async function ensurePrevisoesGroupMatchCardsRendered() {
        const panel = document.getElementById('panel-grupos');
        const container = panel?.querySelector('#groups-scorecards');
        if (!panel || !container) return;
        if (container.dataset.ready === '1') return;

        const stageRows = GROUP_STAGE_ROWS.filter(r => /^[A-L]$/.test(String(r.group ?? '').trim()));

        if (!window.ScoreCards?.renderScoreCardHTML || !window.ScoreCards?.getFlagGetterOnce) {
            container.innerHTML = '';
            container.dataset.ready = '1';
            return;
        }

        const getFlag = await window.ScoreCards.getFlagGetterOnce();
        const stickersCards = stageRows.map((r, i) => window.ScoreCards.renderStickerCard(r, getFlag, i)).join('');
        const scorecards = stageRows.map(r => window.ScoreCards.renderScoreCardHTML(r, getFlag)).join('');
        // <div class="section-title">Figurinhas</div> <div class="section-title">Gráficos Detalhados</div>
        container.innerHTML = `
            <div id="groups-stickers-view" class="groups-display-view">
                <div class="stickers-carousel">
                    <button class="scroll-btn left" onclick="this.nextElementSibling.scrollBy({left: -300, behavior: 'smooth'})">❮</button>
                    <div class="stickers-grid">
                        ${stickersCards}
                    </div>
                    <button class="scroll-btn right" onclick="this.previousElementSibling.scrollBy({left: 300, behavior: 'smooth'})">❯</button>
                </div>
            </div>
            <div id="groups-charts-view" class="groups-display-view">
                <div class="scorecards-grid">
                    ${scorecards}
                </div>
            </div>
        `;

        window.ScoreCards.adjustScoreTotalHeights?.(container);
        container.dataset.ready = '1';
    }

    function buildPrevisoesGroupMatchViews(matchRows) {
        const panel = document.getElementById('panel-grupos');
        if (!panel) return;

        panel.dataset.groupsSource = 'previsoes';
        if (!panel.dataset.groupsMode) panel.dataset.groupsMode = 'stickers';

        ensureGroupsPhaseShell(panel, true);

        let grid = document.getElementById('gg');
        if (grid) grid.style.display = 'none';

        if (!panel.querySelector('#groups-scorecards')) {
            panel.insertAdjacentHTML('beforeend', `<div id="groups-scorecards"></div>`);
        }

        const groups = getGroupLetters(matchRows);
        populateGroupDropdown(groups);
        populateGroupsDateDropdown(matchRows);
        attachGroupFilters();
        setGroupsMode(panel.dataset.groupsMode || 'stickers');
    }

    async function setGroupsMode(mode) {
        const panel = document.getElementById('panel-grupos');
        if (!panel) return;

        // A página de Chances usa apenas a visualização de grupos.
        if (panel.dataset.groupsSource === 'chances') {
            panel.dataset.groupsMode = 'games';
            applyGroupFilters(panel);
            return;
        }

        panel.dataset.groupsMode = mode;

        const filterbar = panel.querySelector('.groups-filterbar');
        if (filterbar) filterbar.classList.add('has-date');

        const probNoteText = panel.querySelector('.groups-prob-note-text');
        if (probNoteText) {
            probNoteText.textContent = '% = Probabilidade estimada de cada placar. Clique para visualizar.';
        }

        panel.querySelectorAll('.groups-mode-btn').forEach(b => {
            b.classList.toggle('active', b.dataset.mode === mode);
        });

        await ensurePrevisoesGroupMatchCardsRendered();

        const stickersView = panel.querySelector('#groups-stickers-view');
        const chartsView = panel.querySelector('#groups-charts-view');

        if (stickersView) stickersView.style.display = (mode === 'stickers') ? '' : 'none';
        if (chartsView) chartsView.style.display = (mode === 'charts') ? '' : 'none';

        applyGroupFilters(panel);
    }

    window.PrevisoesActions = window.PrevisoesActions || {};
    window.PrevisoesActions.setGroupsMode = setGroupsMode;

    // Ponto de entrada da fase de grupos: carrega o CSV e chama a montagem correspondente.
    async function renderGroupStage() {
        const panel = document.getElementById('panel-grupos');
        if (!panel) return;

        try {
            const [matchRows, summaryRows] = await Promise.all([
                loadCSV(MATCHES_CSV_URL),
                loadCSV(GROUPS_PROBS)
            ]);

            GROUP_STAGE_ROWS = matchRows;

            if (panel.dataset.groupsSource === 'chances') {
                buildChancesGroupCards(matchRows, summaryRows);
            } else {
                buildPrevisoesGroupMatchViews(matchRows);
            }
        } catch (e) {
            const grid = document.getElementById('gg');
            if (grid) {
                grid.innerHTML = `
                    <div class="g-card">
                        <div class="g-head">Fase de grupos<span>erro</span></div>
                        <div class="g-team"><div class="g-name">Erro ao carregar ${MATCHES_CSV_URL} ou ${GROUPS_PROBS}.</div></div>
                    </div>
                `;
            }
            console.error(e);
        }
    }

    document.addEventListener('DOMContentLoaded', renderGroupStage);
})();


// ════════════════════════════════════════
// SCORE PROBABILITY CARDS
// Uses partidas.csv
// one_zero = home 1 x 0 away
// ════════════════════════════════════════


(function () {
    function getOutcomeGroups(row) {
        const homeWin = [];
        const draw = [];
        const awayWin = [];

        SCORES.forEach(({ h, a, key, label }) => {
            const value = parseNumber(row[key]);

            if (value === null) return;

            const item = {
                key,
                label,
                homeGoals: h,
                awayGoals: a,
                value
            };

            if (h > a) homeWin.push(item);
            else if (h === a) draw.push(item);
            else awayWin.push(item);
        });

        const byProbability = (a, b) => b.value - a.value;

        homeWin.sort(byProbability);
        draw.sort(byProbability);
        awayWin.sort(byProbability);

        return { homeWin, draw, awayWin };
    }

    function sumOutcomes(outcomes) {
        return outcomes.reduce((sum, item) => sum + item.value, 0); 
    }

    function getOutcomeTotalFromColumn(row, columnName, fallbackOutcomes) {
        const value = parseNumber(row[columnName]);

        if (value === null) {
            return sumOutcomes(fallbackOutcomes);
        }

        return value <= 1 ? value * 100 : value;
    }

    function getBestScore(...groups) {
        return groups
            .flat()
            .slice()
            .sort((a, b) => b.value - a.value)[0];
    }

    function renderOutcomeRows(outcomes) {
        const maxValue = Math.max(...outcomes.map(item => item.value), 1);

        return outcomes.map(item => {
            const width = Math.max(2, (item.value / maxValue) * 100);

            return `
                <div class="score-row">
                    <div class="score-label">${escapeHTML(item.label)}</div>
                    <div class="score-bar-space">
                        <div class="score-bar" style="--w:${width}%"></div>
                    </div>
                    <div class="score-value">${formatPct(item.value)}%</div>
                </div>
            `;
        }).join('');
    }

    function renderOutcomeCard(type, title, totalLabel, totalValue, outcomes) {
        return `
            <div class="score-outcome-card ${type}">
                <div class="score-outcome-title">${escapeHTML(title)}</div>

                <div class="score-outcome-list">
                    ${renderOutcomeRows(outcomes)}
                </div>

                <div class="score-total">
                    <span class="score-total-label">${escapeHTML(totalLabel)}</span>
                    <span class="score-total-value">${formatPct(totalValue)}%</span>
                </div>
            </div>
        `;
    }

    function renderFlag(country, flagUrl) {
        if (flagUrl) {
            return `<img src="${escapeHTML(flagUrl)}" alt="${escapeHTML(country)}">`;
        }

        return `<div class="score-flag-fallback">${escapeHTML(country)}</div>`;
    }

    function renderScoreCard(row, getFlag) {
        const homeCountry = getHomeCountry(row);
        const awayCountry = getAwayCountry(row);

        const { homeWin, draw, awayWin } = getOutcomeGroups(row);

        const best = getBestScore(homeWin, draw, awayWin) || {
            label: '0x0',
            homeGoals: 0,
            awayGoals: 0,
            value: 0
        };

        const rawHomeTotal = getOutcomeTotalFromColumn(row, 'home_win', homeWin);
        const rawDrawTotal = getOutcomeTotalFromColumn(row, 'draw', draw);
        const rawAwayTotal = getOutcomeTotalFromColumn(row, 'away_win', awayWin);

        const homeTotal = Number(rawHomeTotal.toFixed(1));
        const drawTotal = Number(rawDrawTotal.toFixed(1));
        const awayTotal = Number((100 - homeTotal - drawTotal).toFixed(1));

        const homeFlag = getFlag(homeCountry);
        const awayFlag = getFlag(awayCountry);

        const matchTitle = `${homeCountry} X ${awayCountry}`;
        const matchDate = row.date || '';

        const homeReal = row.home_real || '—';
        const awayReal = row.away_real || '—';

        const searchText = normalizeName(`${homeCountry} ${awayCountry}`);

        return `
            <section 
                class="match-card g-card"
                data-home="${escapeHTML(homeCountry)}"
                data-away="${escapeHTML(awayCountry)}"
                data-search="${escapeHTML(searchText)}"
                data-date="${escapeHTML(matchDate)}"
                data-group="${escapeHTML(getMatchGroup(row))}"
            >
                <div class="g-head">
                    <div>${escapeHTML(matchTitle)}</div>
                    <span>${escapeHTML(matchDate)}</span>
                </div>

                <article class="score-card">
                    <div class="score-card-top">
                        <div class="score-flag-wrap">
                            <div class="score-flag">
                                ${renderFlag(homeCountry, homeFlag)}
                            </div>
                        </div>

                        <div class="score-main">
                            <div class="score-main-result">${escapeHTML(best.label.replace('x', ' x '))}</div>
                            <div class="score-main-prob">Probabilidade: ${formatPct(best.value)}%</div>
                        </div>

                        <div class="score-flag-wrap">
                            <div class="score-flag">
                                ${renderFlag(awayCountry, awayFlag)}
                            </div>
                        </div>
                    </div>

                    <div class="score-columns">
                        ${renderOutcomeCard(
                            'home',
                            `${homeCountry} vence`,
                            homeCountry,
                            homeTotal,
                            homeWin
                        )}

                        ${renderOutcomeCard(
                            'draw',
                            'Empate',
                            'Empate',
                            drawTotal,
                            draw
                        )}

                        ${renderOutcomeCard(
                            'away',
                            `${awayCountry} vence`,
                            awayCountry,
                            awayTotal,
                            awayWin
                        )}
                    </div>
                </article>

                <div class="real-result">
                    Resultado Real: <span>${escapeHTML(homeReal)} x ${escapeHTML(awayReal)}</span>
                </div>
            </section>
        `;
    }

    function renderStickerCard(row, getFlag, index = 0) {
        const homeCountry = getHomeCountry(row);
        const awayCountry = getAwayCountry(row);
        const { homeWin, draw, awayWin } = getOutcomeGroups(row);
        const best = getBestScore(homeWin, draw, awayWin) || { label: '0x0', homeGoals: 0, awayGoals: 0, value: 0 };
        
        const rawHomeTotal = getOutcomeTotalFromColumn(row, 'home_win', homeWin);
        const rawDrawTotal = getOutcomeTotalFromColumn(row, 'draw', draw);

        const homeTotal = Number(rawHomeTotal.toFixed(1));
        const drawTotal = Number(rawDrawTotal.toFixed(1));
        const awayTotal = Number((100 - homeTotal - drawTotal).toFixed(1));

        const homeFlag = getFlag(homeCountry);
        const awayFlag = getFlag(awayCountry);
        const matchDate = row.date || '';
        let matchGroup = getMatchGroup(row);
        if (matchGroup && matchGroup.length === 1) {
            matchGroup = 'Grupo ' + matchGroup;
        }
        const searchText = normalizeName(`${homeCountry} ${awayCountry}`);
        
        const randomRot = (Math.random() * 10 - 5).toFixed(1); // -5 to +5 degrees
        const randomY = (Math.random() * 16 - 8).toFixed(1);   // -8 to +8 px

        return `
            <div class="match-card sticker-wrapper"
                onclick="if(window.openStickerModal) window.openStickerModal(this, '${escapeHTML(homeCountry)}', '${escapeHTML(awayCountry)}')"
                style="--rand-rot: ${randomRot}deg; --rand-y: ${randomY}px; cursor: pointer;"
                data-home="${escapeHTML(homeCountry)}"
                data-away="${escapeHTML(awayCountry)}"
                data-search="${escapeHTML(searchText)}"
                data-date="${escapeHTML(matchDate)}"
                data-group="${escapeHTML(getMatchGroup(row))}">
                ${renderSticker(row, best, homeFlag, awayFlag, homeTotal, drawTotal, awayTotal, homeCountry, awayCountry, matchDate, matchGroup)}
            </div>
        `;
    }

    function renderSticker(row, best, homeFlag, awayFlag, homeTotal, drawTotal, awayTotal, homeCountry, awayCountry, matchDate, matchGroup) {
        let gridHtml = '';
        let maxProb = 0;
        
        for (let a = 4; a >= 0; a--) {
            for (let h = 4; h >= 0; h--) {
                const key = `${NUMBER_WORDS[h]}_${NUMBER_WORDS[a]}`;
                const val = parseNumber(row[key]) || 0;
                if (val > maxProb) maxProb = val;
            }
        }
        maxProb = Math.max(maxProb, 1);
        
        for (let h = 4; h >= 0; h--) {
            for (let a = 0; a <= 4; a++) {
                const key = `${NUMBER_WORDS[h]}_${NUMBER_WORDS[a]}`;
                const prob = parseNumber(row[key]) || 0;
                const alpha = Math.min(1, (prob / maxProb) * 0.7 + 0.1);
                const isBest = (h === best.homeGoals && a === best.awayGoals);
                const bg = isBest ? 'rgba(255, 255, 255, 0.95)' : (prob === 0 ? 'rgba(255, 255, 255, 0.04)' : `rgba(255, 255, 255, ${alpha})`);
                const probText = prob < 0.1 ? (prob > 0 ? '&lt;0.1%' : '0%') : `${formatPct(prob)}%`;
                
                gridHtml += `
                    <div class="heatmap-cell ${isBest ? 'best' : ''}" style="background: ${bg}">
                        <div class="prob">${probText}</div>
                        <div class="score">${h}x${a}</div>
                    </div>
                `;
            }
        }

        const maxGoals = Math.max(best.homeGoals, best.awayGoals);
        let borderClass = '';
        if (maxGoals === 2) {
            borderClass = 'border-orange';
        } else if (maxGoals >= 3) {
            borderClass = 'border-purple';
        }

        return `
        <div class="sticker-container ${borderClass}">
            <div class="sticker-bg-blur">
                <div class="blur-flag" style="background-image: url('${homeFlag}')"></div>
                <div class="blur-flag" style="background-image: url('${awayFlag}')"></div>
            </div>
            
            <div class="sticker-glass">
                <div class="sticker-top-info">
                    <span class="info-group">${escapeHTML(matchGroup)}</span>
                    ${matchDate ? `<span class="info-dot">•</span><span class="info-date">${escapeHTML(matchDate)}</span>` : ''}
                </div>
                
                <div class="sticker-header">
                    <div class="team">
                        <div class="sticker-flag">
                            ${renderFlag(homeCountry, homeFlag)}
                        </div>
                        <div class="team-name">${escapeHTML(homeCountry)}</div>
                    </div>
                    
                    <div class="score-center">
                        <div class="most-likely">${escapeHTML(best.label.replace('x', ' - '))}</div>
                        <div class="most-likely-prob">${formatPct(best.value)}%</div>
                    </div>
                    
                    <div class="team">
                        <div class="sticker-flag">
                            ${renderFlag(awayCountry, awayFlag)}
                        </div>
                        <div class="team-name">${escapeHTML(awayCountry)}</div>
                    </div>
                </div>
                
                <div class="heatmap-wrapper">
                    <div class="heatmap-grid">
                        ${gridHtml}
                    </div>
                </div>
                
                <div class="sticker-footer">
                    <div class="footer-bar-container">
                        <div class="f-bar home" style="width: ${homeTotal}%"></div>
                        <div class="f-bar draw" style="width: ${drawTotal}%"></div>
                        <div class="f-bar away" style="width: ${awayTotal}%"></div>
                    </div>
                    <div class="footer-stats-row">
                        <div class="f-stat home">${formatPct(homeTotal)}%</div>
                        <div class="f-stat draw">${formatPct(drawTotal)}%</div>
                        <div class="f-stat away">${formatPct(awayTotal)}%</div>
                    </div>
                </div>
            </div>
        </div>
        `;
    }

    let __flagRowsPromise = null;
    let __flagGetterPromise = null;
    function getFlagRowsOnce() {
    if (!__flagRowsPromise) __flagRowsPromise = loadCSV(FLAGS_CSV_URL);
    return __flagRowsPromise;
    }

    function getFlagGetterOnce() {
    if (!__flagGetterPromise) {
        __flagGetterPromise = getFlagRowsOnce().then(flagRows => {
            const flagMap = new Map(
              flagRows.map(row => [normalizeName(row.country_pt), normalizeFlagUrl(row.svg_github)])
            );

            const getFlag = country => flagMap.get(normalizeName(country)) || '';
            return getFlag;
        });
    }
    return __flagGetterPromise;
    }

    window.ScoreCards = window.ScoreCards || {};
    window.ScoreCards.renderScoreCardHTML = renderScoreCard;
    window.ScoreCards.renderStickerCard = renderStickerCard;
    window.ScoreCards.getFlagRowsOnce = getFlagRowsOnce;
    window.ScoreCards.getFlagGetterOnce = getFlagGetterOnce;
    window.ScoreCards.adjustScoreTotalHeights = adjustScoreTotalHeights;
    
    function renderScorePanelShell(panel, stage) {
        const filtersHTML = stage.showFilters === false ? '' : `
            <div class="filterbar">
                <div class="filter-field search-field">
                    <div class="search-wrap">
                        <span class="search-icon">🔎</span>
                        <input 
                            id="${stage.panelId}-country-filter"
                            class="country-filter" 
                            type="text" 
                            placeholder="Buscar Seleção..."
                            autocomplete="off"
                        >
                    </div>
                </div>

                <div class="filter-field date-field">
                    <div class="date-dropdown">
                        <button type="button" class="date-dropdown-btn">
                            Todas as datas
                            <span>▾</span>
                        </button>

                        <div class="date-dropdown-menu"></div>
                    </div>
                </div>
            </div>
        `;

        panel.innerHTML = `
            ${filtersHTML}

            <div class="stage-content-wrapper"></div>

            <div class="no-results">
                Nenhum confronto encontrado com os filtros selecionados.
            </div>
        `;
    }

    function populateDateFilter(panel) {
        const dateMenu = panel.querySelector('.date-dropdown-menu');
        if (!dateMenu) return;

        const dates = [...panel.querySelectorAll('.match-card')]
            .map(card => card.dataset.date)
            .filter(Boolean);

        const uniqueDates = [...new Set(dates)];

        dateMenu.innerHTML = uniqueDates
            .map(date => `
                <label class="date-option">
                    <input type="checkbox" value="${escapeHTML(date)}">
                    <span>${escapeHTML(date)}</span>
                </label>
            `)
            .join('');
    }

    function attachScoreFilters(panel) {
        installFilterDelegationOnce();
    }

    function adjustScoreTotalHeights(panel) {
        panel.querySelectorAll('.score-total').forEach(total => {
            const label = total.querySelector('.score-total-label');
            if (!label) return;

            const textLength = label.textContent.trim().length;

            total.classList.remove('tall', 'tallest');
            if (textLength > 20) {
                total.classList.add('tallest');
            } else if (textLength > 10) {
                total.classList.add('tall');
            }
        });
    }

    function renderScoreStage(stage, matchRows, getFlag) {
        const stageRows = matchRows.filter(row =>
            normalizeName(getMatchGroup(row)) === normalizeName(stage.groupValue)
        );

        // Se não houver partidas → mostrar placeholder automaticamente
        if (!stageRows.length) {
            renderPlaceholder(
                stage.panelId,
                PLACEHOLDERS[stage.panelId] || 'desta fase'
            );
            return;
        }

        const panel = document.getElementById(stage.panelId);
        if (!panel) return;

        renderScorePanelShell(panel, stage);

        const container = panel.querySelector('.stage-content-wrapper');
        const stickersCards = stageRows.map((row, i) => window.ScoreCards.renderStickerCard(row, getFlag, i)).join('');
        const scorecards = stageRows.map(row => window.ScoreCards.renderScoreCardHTML(row, getFlag)).join('');

        container.innerHTML = stickersCards || scorecards ? `
            <div class="section-title">Figurinhas</div>
            <div class="stickers-carousel">
                <button class="scroll-btn left" onclick="this.nextElementSibling.scrollBy({left: -300, behavior: 'smooth'})">❮</button>
                <div class="stickers-grid">
                    ${stickersCards}
                </div>
                <button class="scroll-btn right" onclick="this.previousElementSibling.scrollBy({left: 300, behavior: 'smooth'})">❯</button>
            </div>
            <div class="section-title">Gráficos Detalhados</div>
            <div class="${stage.gridClass || 'scorecards-grid'}">
                ${scorecards}
            </div>
        ` : `
            <div class="score-empty">
                Nenhum confronto encontrado para esta fase.
            </div>
        `;

        if (stage.showFilters !== false) {
            populateDateFilter(panel);
            attachScoreFilters(panel);
        }
        adjustScoreTotalHeights(panel);
    }

    async function renderScoreStagePanels() {
        try {
            const [matchRows, getFlag] = await Promise.all([
                loadCSV(MATCHES_CSV_URL),
                getFlagGetterOnce()
            ]);
            
            window.ScoreCards.matchesData = matchRows;

            SCORE_STAGES.forEach(stage => {
                renderScoreStage(stage, matchRows, getFlag);
            });
        } catch (error) {
            SCORE_STAGES.forEach(stage => {
                const panel = document.getElementById(stage.panelId);
                if (!panel) return;

                panel.innerHTML = `
                    <div class="score-error">
                        Erro ao carregar os arquivos CSV.<br>
                        Verifique:<br>
                        ${escapeHTML(MATCHES_CSV_URL)}<br>
                        ${escapeHTML(FLAGS_CSV_URL)}
                    </div>
                `;
            });

            console.error(error);
        }
    }

    document.addEventListener('DOMContentLoaded', renderScoreStagePanels);
    
    // Check for sticker deep link after everything is initialized
    window.addEventListener('load', () => {
        setTimeout(async () => {
            const urlParams = new URLSearchParams(window.location.search);
            const stickerParam = urlParams.get('sticker');
            if (stickerParam) {
                const [home, away] = stickerParam.split('-');
                if (home && away && window.openStickerFromData) {
                    window.openStickerFromData(home, away);
                }
            }
        }, 1000);
    });
})();

// Global Modal Functions for Stickers
window.openStickerModal = function(element, home, away) {
    let modal = document.getElementById('sticker-modal');
    if (!modal) {
        modal = document.createElement('div');
        modal.id = 'sticker-modal';
        modal.className = 'sticker-modal';
        modal.innerHTML = `
            <div class="sticker-modal-overlay" onclick="closeStickerModal()"></div>
            <div class="sticker-modal-content">
                <div id="sticker-modal-card-container"></div>
                <div class="sticker-modal-actions">
                    <button class="sticker-modal-btn" onclick="copyStickerLink('${home}', '${away}')">🔗 Compartilhar</button>
                    <button class="sticker-modal-btn" onclick="closeStickerModal()">✖ Fechar</button>
                </div>
            </div>
        `;
        document.body.appendChild(modal);
    } else {
        const shareBtn = modal.querySelector('.sticker-modal-btn');
        shareBtn.setAttribute('onclick', `copyStickerLink('${home}', '${away}')`);
    }

    const container = modal.querySelector('#sticker-modal-card-container');
    container.innerHTML = '';
    
    if (element) {
        const sticker = element.querySelector('.sticker-container').cloneNode(true);
        // Reset transformations on the clone so it displays perfectly in the modal
        sticker.style.transform = 'none';
        container.appendChild(sticker);
    }

    modal.style.display = 'flex';
    setTimeout(() => modal.classList.add('active'), 10);
};

window.openStickerFromData = async function(home, away) {
    if (!window.ScoreCards || !window.ScoreCards.matchesData) return;
    const row = window.ScoreCards.matchesData.find(r => r.home_team === home && r.away_team === away);
    if (!row) return;

    const getFlag = await window.ScoreCards.getFlagGetterOnce();
    const temp = document.createElement('div');
    temp.innerHTML = window.ScoreCards.renderStickerCard(row, getFlag);
    const card = temp.firstElementChild;
    window.openStickerModal(card, home, away);
};

window.closeStickerModal = function() {
    const modal = document.getElementById('sticker-modal');
    if (modal) {
        modal.classList.remove('active');
        setTimeout(() => modal.style.display = 'none', 300);
    }
};

window.copyStickerLink = function(home, away) {
    const url = new URL(window.location.href);
    url.searchParams.set('sticker', `${home}-${away}`);
    navigator.clipboard.writeText(url.toString()).then(() => {
        showToast('O link foi copiado para a área de transferência!');
    });
};

function showToast(msg) {
    let toast = document.getElementById('sticker-toast');
    if (!toast) {
        toast = document.createElement('div');
        toast.id = 'sticker-toast';
        toast.className = 'toast-message';
        document.body.appendChild(toast);
    }
    toast.textContent = msg;
    toast.classList.add('show');
    setTimeout(() => toast.classList.remove('show'), 3000);
}

function abrirTabPelaURL() {
    const tabName = window.location.hash.replace('#', '').trim();

    if (!tabName) return;

    const tabsValidas = ['bracket', 'grupos', 'r32', 'oitavas', 'quartas', 'semis', 'final'];

    if (!tabsValidas.includes(tabName)) return;

    const botao = [...document.querySelectorAll('.tabs .tab')]
        .find(btn => {
            const onclick = btn.getAttribute('onclick') || '';
            return onclick.includes(`'${tabName}'`) || onclick.includes(`"${tabName}"`);
        });

    if (botao && typeof switchTab === 'function') {
        switchTab(tabName, botao);

        document.querySelector('.dash')?.scrollIntoView({
            behavior: 'smooth',
            block: 'start'
        });
    }
}

window.addEventListener('DOMContentLoaded', abrirTabPelaURL);
window.addEventListener('hashchange', abrirTabPelaURL);