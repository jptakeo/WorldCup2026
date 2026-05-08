import json
import os

# Standalone D3 template used by generated dashboard files.
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang='pt-br'>
<head>
    <meta charset='UTF-8'>
    <meta name='viewport' content='width=device-width, initial-scale=1.0'>
    <title>Dashboard Copa</title>
    <script src='https://d3js.org/d3.v7.min.js'></script>
    <style>
        body { background-color: #0d1117; color: #c9d1d9; font-family: sans-serif; margin: 0; overflow: hidden; height: 100vh; }
        svg { cursor: grab; } svg:active { cursor: grabbing; }
        .link { fill: none; stroke: #30363d; stroke-width: 2px; opacity: 0.8; }
        .visual-node { fill: #0d1117; stroke: #58a6ff; stroke-width: 2.5px; pointer-events: none; }
        .has-children { fill: #58a6ff !important; }
        .node text { font-size: 14px; fill: #f0f6fc; pointer-events: none; text-shadow: 1px 1px 2px rgba(0,0,0,0.9); }
        .hitbox { fill: transparent; cursor: pointer; }
        image { pointer-events: none; }
        
        /* Floating title overlay; pointer events pass through to the graph. */
        #title-box {
            position: absolute;
            top: 20px;
            left: 20px;
            z-index: 10;
            pointer-events: none;
        }
        #title-box h1 { margin: 0; font-size: 24px; color: #f0f6fc; }
        #title-box h3 { margin: 5px 0 0 0; font-size: 14px; color: #8b949e; font-weight: normal; }
    </style>
</head>
<body>
    <!-- Floating title box -->
    <div id='title-box'>
        <h1>{{TITULO_COPA}}</h1>
        <h3>Modelo: {{NOME_MODELO}}</h3>
    </div>

    <div id='graph-container'></div>
    <script>
        const treeData = __JSON_DATA__;
        const margin = {top: 20, right: 350, bottom: 20, left: 150};
        const svg = d3.select('#graph-container').append('svg')
            .attr('width', window.innerWidth).attr('height', window.innerHeight)
            .call(d3.zoom().scaleExtent([0.1, 4]).on('zoom', e => g.attr('transform', e.transform)))
            .append('g').attr('transform', `translate(${margin.left},${window.innerHeight / 2})`);

        const g = svg;
        const tree = d3.tree().nodeSize([35, 300]);
        const root = d3.hierarchy(treeData, d => d.children);
        root.x0 = 0; root.y0 = 0;

        if (root.children) { root.children.forEach(collapse); }

        function collapse(d) {
            if (d.children) {
                d._children = d.children;
                d._children.forEach(collapse);
                d.children = null;
            }
        }

        function update(source) {
            const treeData = tree(root);
            const nodes = treeData.descendants(), links = treeData.descendants().slice(1);
            nodes.forEach(d => d.y = d.depth * 300);

            const node = g.selectAll('g.node').data(nodes, d => d.id || (d.id = ++i));

            const nodeEnter = node.enter().append('g').attr('class', 'node')
                .attr('transform', d => `translate(${source.y0},${source.x0})`);

            nodeEnter.append('circle').attr('r', 12).attr('class', 'hitbox').on('click', (e, d) => {
                if (d.children) { d._children = d.children; d.children = null; }
                else { d.children = d._children; d._children = null; }
                update(d);
            });

            nodeEnter.append('circle').attr('r', 6).attr('class', 'visual-node')
                .classed('has-children', d => d._children || d.children);

            nodeEnter.filter(d => d.data.icon).append('image')
                .attr('href', d => d.data.icon).attr('x', 15).attr('y', -10).attr('width', 20).attr('height', 20);

            nodeEnter.append('text').attr('dy', '.35em').attr('x', d => d.data.icon ? 40 : 15).text(d => d.data.name);

            const nodeUpdate = nodeEnter.merge(node);
            nodeUpdate.transition().duration(400).attr('transform', d => `translate(${d.y},${d.x})`);
            nodeUpdate.select('.visual-node').classed('has-children', d => d._children);

            const nodeExit = node.exit().transition().duration(400)
                .attr('transform', d => `translate(${source.y},${source.x})`).remove();

            const link = g.selectAll('path.link').data(links, d => d.id);
            const linkEnter = link.enter().insert('path', 'g').attr('class', 'link')
                .attr('d', d => { const o = {x: source.x0, y: source.y0}; return diagonal(o, o); });

            const linkUpdate = linkEnter.merge(link);
            linkUpdate.transition().duration(400).attr('d', d => diagonal(d, d.parent));

            link.exit().transition().duration(400).attr('d', d => { const o = {x: source.x, y: source.y}; return diagonal(o, o); }).remove();
            nodes.forEach(d => { d.x0 = d.x; d.y0 = d.y; });
        }

        function diagonal(s, d) { return `M ${s.y} ${s.x} C ${(s.y + d.y) / 2} ${s.x}, ${(s.y + d.y) / 2} ${d.x}, ${d.y} ${d.x}`; }

        let i = 0; update(root);
    </script>
</body>
</html>"""

# ISO/flagcdn codes used in generated dashboards.
ISO_FLAGS = {
    "Argentina": "ar",
    "Brazil": "br",
    "Ecuador": "ec",
    "Uruguay": "uy",
    "Colombia": "co",
    "Paraguay": "py",
    "Spain": "es",
    "Switzerland": "ch",
    "England": "gb-eng",
    "Germany": "de",
    "France": "fr",
    "Netherlands": "nl",
    "Croatia": "hr",
    "Portugal": "pt",
    "Belgium": "be",
    "Austria": "at",
    "Norway": "no",
    "Czech Republic": "cz",
    "Scotland": "gb-sct",
    "Turkey": "tr",
    "Sweden": "se",
    "Bosnia and Herzegovina": "ba",
    "Morocco": "ma",
    "Senegal": "sn",
    "Ivory Coast": "ci",
    "Algeria": "dz",
    "Egypt": "eg",
    "Tunisia": "tn",
    "South Africa": "za",
    "DR Congo": "cd",
    "Ghana": "gh",
    "Cape Verde": "cv",
    "Japan": "jp",
    "South Korea": "kr",
    "Iran": "ir",
    "Uzbekistan": "uz",
    "Qatar": "qa",
    "Saudi Arabia": "sa",
    "Iraq": "iq",
    "Jordan": "jo",
    "Australia": "au",
    "Canada": "ca",
    "Mexico": "mx",
    "United States": "us",
    "Panama": "pa",
    "Haiti": "ht",
    "Curaçao": "cw",
    "New Zealand": "nz",
    "Poland": "pl",
    "Denmark": "dk",
    "Serbia": "rs",
    "Cameroon": "cm",
    "Costa Rica": "cr",
    "Wales": "gb-wls",
}


def generate_dashboard(
    json_file, output_file, fases_nome, participantes, chunk_size, title, nome_modelo
):
    """
    Gera o dashboard HTML genérico para qualquer configuração de Copa,
    incluindo o título da simulação e o modelo estatístico utilizado.
    """
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"Arquivo {json_file} não encontrado.")

    with open(json_file, "r") as f:
        data = json.load(f)

    root = {"name": title, "children": []}
    fases_exibir = list(fases_nome.keys())

    for key in fases_exibir:
        node = {"name": fases_nome.get(key, key), "children": []}

        # Show only tournament participants, ordered by probability.
        team_probs = [x for x in data.get(key, []) if x["team"] in participantes]
        team_probs = sorted(team_probs, key=lambda x: x["probability"], reverse=True)

        # Split large rankings into readable expandable groups.
        for i in range(0, len(team_probs), chunk_size):
            chunk = team_probs[i : i + chunk_size]
            if not chunk or chunk[0]["probability"] == 0:
                continue

            aba = {"name": f"Top {i+len(chunk)}", "children": []}
            for item in chunk:
                flag = ISO_FLAGS.get(item["team"], "")
                aba["children"].append(
                    {
                        "name": f"{item['team']}: {item['probability']*100:.2f}%",
                        "icon": f"https://flagcdn.com/w40/{flag}.png" if flag else "",
                    }
                )
            node["children"].append(aba)

        root["children"].append(node)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    html_final = HTML_TEMPLATE
    html_final = html_final.replace("__JSON_DATA__", json.dumps(root))
    html_final = html_final.replace("{{TITULO_COPA}}", title)
    html_final = html_final.replace("{{NOME_MODELO}}", nome_modelo)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_final)

    print(f"Dashboard '{title}' gerado com sucesso em: {output_file}")
