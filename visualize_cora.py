# Cora Citation Network visualization
# Downloads the dataset, builds a graph with NetworkX, and renders it as an
# interactive HTML file using PyVis. Run this script and open the output HTML.

import os
import io
import ssl
import tarfile
import urllib.error
import urllib.request
import collections

import networkx as nx
from pyvis.network import Network


CORA_URL    = "https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz"
DATA_DIR    = "cora_data"
OUTPUT_HTML = "cora_visualization.html"

# one color per topic so clusters are easy to spot
TOPIC_COLORS = {
    "Case_Based":             "#E74C3C",
    "Genetic_Algorithms":     "#F39C12",
    "Neural_Networks":        "#2ECC71",
    "Probabilistic_Methods":  "#9B59B6",
    "Reinforcement_Learning": "#3498DB",
    "Rule_Learning":          "#1ABC9C",
    "Theory":                 "#E67E22",
}

STABILIZATION_ITERATIONS = 150


def download_cora():
    # grab the dataset if we don't already have it
    content_path = os.path.join(DATA_DIR, "cora", "cora.content")
    cites_path   = os.path.join(DATA_DIR, "cora", "cora.cites")

    if os.path.exists(content_path) and os.path.exists(cites_path):
        print("  Dataset already downloaded, skipping.")
        return content_path, cites_path

    print(f"  Downloading from: {CORA_URL}")
    try:
        # macOS sometimes rejects the SSL cert, so fall back to unverified
        try:
            with urllib.request.urlopen(CORA_URL) as r:
                data = r.read()
        except urllib.error.URLError:
            ctx = ssl._create_unverified_context()
            with urllib.request.urlopen(CORA_URL, context=ctx) as r:
                data = r.read()
    except Exception as e:
        raise RuntimeError(f"Download failed: {e}") from e

    print("  Extracting...")
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        try:
            tar.extractall(DATA_DIR, filter="data")   # Python 3.12+
        except TypeError:
            tar.extractall(DATA_DIR)                  # Python 3.9-3.11

    return content_path, cites_path


def build_graph(content_path, cites_path):
    # read the two data files and build an undirected graph
    # cora.content: paper_id, 1433 word features, topic label
    # cora.cites:   citing paper -> cited paper
    G = nx.Graph()

    print("  Loading nodes...")
    with open(content_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            paper_id = parts[0]
            topic    = parts[-1]
            color    = TOPIC_COLORS.get(topic, "#95A5A6")
            G.add_node(paper_id, topic=topic, color=color)

    print("  Loading edges...")
    skipped = 0
    with open(cites_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 2:
                continue
            src, dst = parts[0], parts[1]
            if src in G and dst in G:
                G.add_edge(src, dst)
            else:
                skipped += 1

    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges ({skipped} skipped)\n")
    return G


def build_pyvis(G):
    # set up the PyVis network with Barnes-Hut physics
    # nodes are sized by degree and colored by topic
    net = Network(
        height="100vh",
        width="100%",
        bgcolor="#0d1117",
        font_color="#c9d1d9",
        notebook=False,
        cdn_resources="remote",
    )

    net.set_options(f"""
    {{
      "physics": {{
        "enabled": true,
        "solver": "barnesHut",
        "barnesHut": {{
          "theta": 0.5,
          "gravitationalConstant": -10000,
          "centralGravity": 0.05,
          "springLength": 150,
          "springConstant": 0.04,
          "damping": 0.4,
          "avoidOverlap": 0
        }},
        "timestep": 0.8,
        "stabilization": {{
          "enabled": true,
          "iterations": {STABILIZATION_ITERATIONS},
          "updateInterval": 25,
          "fit": true
        }}
      }},
      "interaction": {{
        "hover": true,
        "tooltipDelay": 80,
        "zoomView": true,
        "dragView": true,
        "dragNodes": true,
        "navigationButtons": false,
        "keyboard": {{ "enabled": true, "bindToWindow": false }}
      }},
      "nodes": {{
        "shape": "dot",
        "borderWidth": 1,
        "borderWidthSelected": 3,
        "shadow": {{ "enabled": true, "size": 8, "x": 0, "y": 0 }},
        "font": {{ "size": 0 }}
      }},
      "edges": {{
        "color": {{
          "color": "#30363d",
          "highlight": "#58a6ff",
          "hover": "#8b949e"
        }},
        "width": 0.6,
        "selectionWidth": 2.5,
        "smooth": {{ "enabled": true, "type": "continuous", "roundness": 0.3 }},
        "shadow": false
      }}
    }}
    """)

    for node_id, attr in G.nodes(data=True):
        degree = G.degree(node_id)
        topic  = attr.get("topic", "Unknown")
        color  = attr.get("color", "#95A5A6")

        # scale size by degree, capped between 6 and 28 px
        size = 6 + min(degree * 1.8, 22)

        net.add_node(
            node_id,
            label="",
            title="",
            color={
                "background": color,
                "border":    "#ffffff22",
                "highlight": {"background": "#ffffff", "border": color},
                "hover":     {"background": color, "border": "#ffffff99"},
            },
            size=size,
            opacity=1.0,
            topic=topic,
            citations=degree,
        )

    for src, dst in G.edges():
        src_topic = G.nodes[src].get("topic", "")
        dst_topic = G.nodes[dst].get("topic", "")
        if src_topic == dst_topic and src_topic in TOPIC_COLORS:
            col = {"color": TOPIC_COLORS[src_topic], "opacity": 0.3}
        else:
            col = {"color": "#30363d", "opacity": 0.7}
        net.add_edge(str(src), str(dst), color=col)

    return net


def inject_interactions_and_legend(html_path, G):
    # PyVis gives us a bare HTML file. This function injects all the extra
    # UI on top: legend, filters, stats panel, loading screen, tooltip,
    # ego-network highlight, and the 3D view toggle.

    topic_counts = collections.Counter(
        attr.get("topic", "Unknown") for _, attr in G.nodes(data=True)
    )
    top_papers = sorted(G.degree(), key=lambda x: x[1], reverse=True)[:5]
    avg_degree = round(sum(d for _, d in G.degree()) / G.number_of_nodes(), 2)
    density    = round(nx.density(G), 5)
    max_degree = max(d for _, d in G.degree())

    with open(html_path, encoding="utf-8") as f:
        html = f.read()

    # legend panel (left side)
    legend_items = "".join(
        f"<div class='topic-filter' data-topic='{t}'"
        f" onclick='toggleTopic(\"{t}\")'  style='"
        f"display:flex;align-items:center;gap:9px;margin:4px 0;"
        f"cursor:pointer;transition:opacity 0.2s;user-select:none;"
        f"border-radius:6px;padding:3px 5px;'"
        f" onmouseover=\"this.style.background='#21262d';\""
        f"  onmouseout=\"this.style.background='transparent';\">"
        f"<span style='width:11px;height:11px;border-radius:50%;"
        f"background:{col};flex-shrink:0;box-shadow:0 0 5px {col}88;'></span>"
        f"<span class='topic-label' style='font-size:12px;color:#c9d1d9;'>"
        f"{t.replace('_', ' ')}</span>"
        f"<span style='margin-left:auto;font-size:10px;color:#444d56;'>"
        f"{topic_counts.get(t, 0)}</span>"
        f"</div>"
        for t, col in TOPIC_COLORS.items()
    )

    legend_div = f"""
    <div id="cora-legend" style="
        position:fixed;top:18px;left:18px;z-index:9999;
        background:#161b22cc;backdrop-filter:blur(8px);
        border:1px solid #30363d;border-radius:12px;
        padding:14px 18px;
        font-family:Segoe UI,Arial,sans-serif;
        box-shadow:0 4px 24px rgba(0,0,0,0.7);
        width:215px;">
      <div id="legend-title" onclick='toggleAllTopics()' title='Click to show / hide all topics'
           style='font-size:13px;font-weight:700;
                  color:#58a6ff;margin-bottom:8px;letter-spacing:0.4px;
                  cursor:pointer;display:flex;align-items:center;
                  justify-content:space-between;
                  border-radius:6px;padding:3px 5px;margin:-3px -5px 8px;
                  transition:background 0.2s;user-select:none;'
           onmouseover="this.style.background='#21262d';"
           onmouseout="this.style.background='transparent';">
        <span>Cora &#8212; Research Topics</span>
        <span id="legend-toggle-icon" style='font-size:11px;color:#8b949e;margin-left:8px;'>&#9679; all</span>
      </div>
      {legend_items}
      <hr style='border:none;border-top:1px solid #30363d;margin:10px 0 8px'/>
      <div style='font-size:11px;font-weight:600;color:#c9d1d9;margin-bottom:6px;'>
        Min Citations Filter
      </div>
      <input type='range' id='degree-slider'
             min='0' max='{max_degree}' value='0' step='1'
             oninput='updateDegreeFilter(this.value)'
             style='width:100%;accent-color:#58a6ff;cursor:pointer;'/>
      <div style='font-size:10px;color:#8b949e;margin-top:4px;text-align:center;'>
        Showing papers with &#8805;
        <b id='degree-val' style='color:#c9d1d9;'>0</b> citations
      </div>
      <button onclick='resetAllFilters()' style='
          margin-top:10px;width:100%;padding:5px 0;
          background:#21262d;border:1px solid #30363d;border-radius:6px;
          color:#8b949e;font-size:11px;cursor:pointer;
          font-family:Segoe UI,Arial,sans-serif;
          transition:background 0.2s,color 0.2s;'
          onmouseover="this.style.background='#30363d';this.style.color='#c9d1d9';"
          onmouseout="this.style.background='#21262d';this.style.color='#8b949e';">
        Reset Filters
      </button>
      <hr style='border:none;border-top:1px solid #30363d;margin:10px 0 8px'/>
      <div style='font-size:10.5px;color:#8b949e;line-height:1.8;'>
        <b style='color:#c9d1d9;'>Click topic</b> &#8594; activate / deactivate<br/>
        <b style='color:#c9d1d9;'>Hover</b> node &#8594; paper details<br/>
        <b style='color:#c9d1d9;'>Click</b> node &#8594; ego-network<br/>
        <b style='color:#c9d1d9;'>Scroll/Drag</b> &#8594; zoom &amp; pan
      </div>
    </div>
    """

    # stats panel (right side)
    top_papers_html = "".join(
        f"<div style='display:flex;justify-content:space-between;"
        f"margin:3px 0;align-items:center;'>"
        f"<span style='color:#c9d1d9;font-family:monospace;font-size:10px;'>#{pid}</span>"
        f"<span style='color:{TOPIC_COLORS.get(G.nodes[pid].get('topic',''), '#95A5A6')};"
        f"font-size:10px;'>{deg} &#9679;</span>"
        f"</div>"
        for pid, deg in top_papers
    )

    stats_panel = f"""
    <div id="stats-panel" style="
        position:fixed;top:18px;right:18px;z-index:9999;
        background:#161b22cc;backdrop-filter:blur(8px);
        border:1px solid #30363d;border-radius:12px;
        padding:14px 18px;
        font-family:Segoe UI,Arial,sans-serif;
        box-shadow:0 4px 24px rgba(0,0,0,0.7);
        width:215px;">
      <div style='font-size:13px;font-weight:700;
                  color:#58a6ff;margin-bottom:10px;letter-spacing:0.4px;'>
        Graph Statistics
      </div>
      <div style='display:grid;grid-template-columns:1fr auto;
                  row-gap:5px;font-size:11px;color:#8b949e;'>
        <span>Nodes</span>
        <span style='color:#c9d1d9;text-align:right;'>{G.number_of_nodes():,}</span>
        <span>Edges</span>
        <span style='color:#c9d1d9;text-align:right;'>{G.number_of_edges():,}</span>
        <span>Avg Degree</span>
        <span style='color:#c9d1d9;text-align:right;'>{avg_degree}</span>
        <span>Density</span>
        <span style='color:#c9d1d9;text-align:right;'>{density}</span>
        <span>Max Degree</span>
        <span style='color:#c9d1d9;text-align:right;'>{max_degree}</span>
      </div>
      <hr style='border:none;border-top:1px solid #30363d;margin:10px 0 8px'/>
      <div style='font-size:11px;font-weight:600;color:#c9d1d9;margin-bottom:6px;'>
        Top Cited Papers
      </div>
      {top_papers_html}
    </div>
    """

    # full-screen loading screen shown while the layout stabilizes
    loading_overlay = f"""
    <div id="loading-overlay" style="
        position:fixed;top:0;left:0;width:100vw;height:100vh;z-index:99998;
        background:#0d1117;
        display:flex;flex-direction:column;align-items:center;justify-content:center;
        font-family:'Segoe UI',Arial,sans-serif;
        overflow:hidden;
        transition:opacity 0.8s ease;">

      <!-- Radial glow backdrop -->
      <div style="position:absolute;top:50%;left:50%;
        width:700px;height:700px;border-radius:50%;
        background:radial-gradient(ellipse,#0d2045 0%,transparent 70%);
        transform:translate(-50%,-50%);pointer-events:none;"></div>

      <!-- Floating topic-colored orbs -->
      <div style="position:absolute;inset:0;pointer-events:none;overflow:hidden;">
        <div style="position:absolute;top:12%;left:8%;width:10px;height:10px;border-radius:50%;background:#E74C3C;box-shadow:0 0 18px #E74C3C;animation:orb-float 7s ease-in-out infinite;opacity:0.7;"></div>
        <div style="position:absolute;top:70%;left:6%;width:8px;height:8px;border-radius:50%;background:#F39C12;box-shadow:0 0 14px #F39C12;animation:orb-float 9s ease-in-out 1s infinite;opacity:0.6;"></div>
        <div style="position:absolute;top:20%;left:88%;width:12px;height:12px;border-radius:50%;background:#2ECC71;box-shadow:0 0 20px #2ECC71;animation:orb-float 8s ease-in-out 0.5s infinite;opacity:0.7;"></div>
        <div style="position:absolute;top:75%;left:85%;width:9px;height:9px;border-radius:50%;background:#9B59B6;box-shadow:0 0 16px #9B59B6;animation:orb-float 11s ease-in-out 2s infinite;opacity:0.6;"></div>
        <div style="position:absolute;top:45%;left:4%;width:7px;height:7px;border-radius:50%;background:#3498DB;box-shadow:0 0 14px #3498DB;animation:orb-float 6s ease-in-out 1.5s infinite;opacity:0.65;"></div>
        <div style="position:absolute;top:85%;left:45%;width:8px;height:8px;border-radius:50%;background:#1ABC9C;box-shadow:0 0 14px #1ABC9C;animation:orb-float 10s ease-in-out 0.8s infinite;opacity:0.6;"></div>
        <div style="position:absolute;top:8%;left:52%;width:10px;height:10px;border-radius:50%;background:#E67E22;box-shadow:0 0 16px #E67E22;animation:orb-float 8.5s ease-in-out 1.2s infinite;opacity:0.65;"></div>
        <div style="position:absolute;top:55%;left:92%;width:6px;height:6px;border-radius:50%;background:#58a6ff;box-shadow:0 0 12px #58a6ff;animation:orb-float 7.5s ease-in-out 2.5s infinite;opacity:0.5;"></div>
        <div style="position:absolute;top:30%;left:15%;width:5px;height:5px;border-radius:50%;background:#2ECC71;box-shadow:0 0 10px #2ECC71;animation:orb-float 12s ease-in-out 3s infinite;opacity:0.4;"></div>
      </div>

      <!-- Main content card -->
      <div style="position:relative;z-index:1;display:flex;flex-direction:column;
                  align-items:center;text-align:center;">

        <!-- Big title -->
        <div style="font-size:72px;font-weight:800;letter-spacing:16px;
                    background:linear-gradient(135deg,#58a6ff 0%,#79c0ff 50%,#a5d6ff 100%);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                    background-clip:text;
                    filter:drop-shadow(0 0 30px #58a6ff55);
                    margin-bottom:6px;line-height:1;">
          CORA
        </div>
        <div style="font-size:12px;letter-spacing:6px;color:#444d56;
                    text-transform:uppercase;margin-bottom:36px;">
          Citation Network Explorer
        </div>

        <!-- Dataset stat pills -->
        <div style="display:flex;gap:14px;margin-bottom:48px;animation:fade-up 0.6s ease 0.3s both;">
          <div style="background:#161b22;border:1px solid #30363d;border-radius:10px;
                      padding:12px 20px;min-width:90px;">
            <div style="font-size:20px;font-weight:700;color:#58a6ff;">2,708</div>
            <div style="font-size:10px;color:#586069;letter-spacing:1px;margin-top:3px;">PAPERS</div>
          </div>
          <div style="background:#161b22;border:1px solid #30363d;border-radius:10px;
                      padding:12px 20px;min-width:90px;">
            <div style="font-size:20px;font-weight:700;color:#2ECC71;">5,278</div>
            <div style="font-size:10px;color:#586069;letter-spacing:1px;margin-top:3px;">CITATIONS</div>
          </div>
          <div style="background:#161b22;border:1px solid #30363d;border-radius:10px;
                      padding:12px 20px;min-width:90px;">
            <div style="font-size:20px;font-weight:700;color:#E67E22;">7</div>
            <div style="font-size:10px;color:#586069;letter-spacing:1px;margin-top:3px;">TOPICS</div>
          </div>
        </div>

        <!-- Progress section -->
        <div style="width:380px;">
          <div style="display:flex;justify-content:space-between;
                      font-size:10px;color:#444d56;letter-spacing:1px;
                      margin-bottom:8px;text-transform:uppercase;">
            <span id="loading-subtitle">Initializing&hellip;</span>
            <span id="loading-progress-text">0 / {STABILIZATION_ITERATIONS}</span>
          </div>
          <div style="width:100%;height:5px;background:#161b22;
                      border-radius:99px;overflow:hidden;border:1px solid #21262d;">
            <div id="loading-bar" style="
              height:100%;width:0%;
              background:linear-gradient(90deg,#1f6feb 0%,#58a6ff 55%,#79c0ff 100%);
              border-radius:99px;
              box-shadow:0 0 10px #58a6ff88;
              transition:width 0.18s ease;"></div>
          </div>

          <!-- Topic color bar (decorative, shows all 7 colors) -->
          <div style="display:flex;gap:3px;margin-top:10px;border-radius:4px;overflow:hidden;opacity:0.5;">
            <div style="flex:298;height:3px;background:#E74C3C;"></div>
            <div style="flex:418;height:3px;background:#F39C12;"></div>
            <div style="flex:818;height:3px;background:#2ECC71;"></div>
            <div style="flex:426;height:3px;background:#9B59B6;"></div>
            <div style="flex:217;height:3px;background:#3498DB;"></div>
            <div style="flex:180;height:3px;background:#1ABC9C;"></div>
            <div style="flex:351;height:3px;background:#E67E22;"></div>
          </div>
        </div>

      </div>
    </div>

    <style>
    @keyframes orb-float {{
      0%,100% {{ transform:translateY(0px) translateX(0px); }}
      33%      {{ transform:translateY(-18px) translateX(8px); }}
      66%      {{ transform:translateY(10px) translateX(-6px); }}
    }}
    @keyframes fade-up {{
      from {{ opacity:0; transform:translateY(16px); }}
      to   {{ opacity:1; transform:translateY(0); }}
    }}
    </style>
    """

    # 2D / 3D toggle button and the container div for the 3D graph
    view_toggle_html = """
    <div id="view-toggle-wrap" style="
        position:fixed;bottom:28px;left:50%;transform:translateX(-50%);
        z-index:9999;display:flex;gap:0;border-radius:20px;overflow:hidden;
        border:1px solid #30363d;box-shadow:0 4px 20px rgba(0,0,0,0.6);
        font-family:'Segoe UI',Arial,sans-serif;">
      <button id="btn-2d" onclick="setViewMode('2d')" style="
          padding:7px 20px;font-size:12px;cursor:pointer;letter-spacing:0.5px;
          background:#58a6ff;color:#0d1117;border:none;font-weight:600;
          transition:all 0.2s;">
        2D
      </button>
      <button id="btn-3d" onclick="setViewMode('3d')" style="
          padding:7px 20px;font-size:12px;cursor:pointer;letter-spacing:0.5px;
          background:#161b22;color:#8b949e;border:none;font-weight:600;
          transition:all 0.2s;">
        3D
      </button>
    </div>

    <!-- 3D graph container (hidden until 3D mode activated) -->
    <div id="graph-3d-container" style="
        display:none;position:fixed;top:0;left:0;
        width:100%;height:100%;
        background:#0d1117;z-index:1000;">
    </div>
    """

    # hover tooltip — reused by both 2D and 3D views
    custom_tooltip = """
    <div id="custom-tooltip" style="
        display:none;position:fixed;z-index:10000;
        background:#161b22;border-radius:9px;
        padding:11px 15px;font-family:Segoe UI,Arial,sans-serif;
        font-size:12px;line-height:1.85;
        min-width:195px;max-width:245px;
        pointer-events:none;
        box-shadow:0 6px 24px rgba(0,0,0,0.7);"></div>

    <script type="text/javascript">
    var tooltipEl = document.getElementById("custom-tooltip");
    var mouseX = 0, mouseY = 0;

    document.addEventListener("mousemove", function(e) {
        mouseX = e.clientX; mouseY = e.clientY;
        if (tooltipEl.style.display === "block") placeTooltip();
    });

    function placeTooltip() {
        var x = mouseX + 18, y = mouseY - 12;
        if (x + 250 > window.innerWidth)  x = mouseX - 255;
        if (y + 130 > window.innerHeight) y = mouseY - 130;
        tooltipEl.style.left = x + "px";
        tooltipEl.style.top  = y + "px";
    }

    network.on("hoverNode", function(params) {
        var node  = nodes.get(params.node);
        var topic = (node.topic || "Unknown").replace(/_/g, " ");
        var deg   = node.citations != null ? node.citations : "–";
        var bgCol = node.color && node.color.background ? node.color.background : "#58a6ff";
        tooltipEl.style.border = "1px solid " + bgCol;
        tooltipEl.innerHTML =
            "<div style='color:" + bgCol + ";font-weight:700;font-size:13px;margin-bottom:6px;'>"
            + "&#9679;&nbsp;" + topic + "</div>"
            + "<hr style='border:none;border-top:1px solid #30363d;margin:0 0 7px'/>"
            + "<div style='color:#8b949e;display:grid;grid-template-columns:90px 1fr;row-gap:3px;'>"
            + "<span>Paper ID</span><span style='color:#c9d1d9;'>" + node.id + "</span>"
            + "<span>Citations</span><span style='color:#c9d1d9;'>" + deg + "</span>"
            + "<span>Topic</span><span style='color:#c9d1d9;'>" + topic + "</span>"
            + "</div>";
        tooltipEl.style.display = "block";
        placeTooltip();
    });

    network.on("blurNode", function() { tooltipEl.style.display = "none"; });
    </script>
    """

    # main JS — filters, highlight, loading progress
    main_js = """
    <script type="text/javascript">

    // filter and highlight state
    var hiddenTopics     = new Set();
    var minCitations     = 0;
    var filteredOutNodes = new Set();
    var isHighlighted    = false;
    var originalEdgeColors = {};

    var DIM_OPACITY = 0.06;
    var ACTIVE_EDGE = { color: "#c9d1d9", opacity: 1.0 };
    var DIMMED_EDGE = { color: "#ffffff", opacity: 0.03 };

    edges.get().forEach(function(e) { originalEdgeColors[e.id] = e.color; });

    // update the loading bar as the layout runs
    network.on("stabilizationProgress", function(params) {
        var pct = (params.iterations / params.total) * 100;
        var bar  = document.getElementById("loading-bar");
        var txt  = document.getElementById("loading-progress-text");
        var sub  = document.getElementById("loading-subtitle");
        if (bar) bar.style.width = pct + "%";
        if (txt) txt.textContent = params.iterations + " / " + params.total + " iterations";
        if (sub && pct > 50) sub.textContent = "Placing nodes\u2026";
        if (sub && pct > 85) sub.textContent = "Almost ready\u2026";
    });

    // layout finished — fill bar to 100% then fade out the overlay
    network.on("stabilizationIterationsDone", function() {
        network.fit({ animation: { duration: 800, easingFunction: "easeInOutQuad" } });

        var bar = document.getElementById("loading-bar");
        if (bar) bar.style.width = "100%";

        setTimeout(function() {
            var overlay = document.getElementById("loading-overlay");
            if (overlay) {
                overlay.style.opacity = "0";
                setTimeout(function() { overlay.style.display = "none"; }, 700);
            }
        }, 350);
    });

    // apply current topic and degree filters to the graph
    function applyFilters() {
        filteredOutNodes.clear();
        isHighlighted = false;
        var nodeUpdates = [], edgeUpdates = [];

        nodes.get().forEach(function(n) {
            var visible = !hiddenTopics.has(n.topic) && (n.citations || 0) >= minCitations;
            if (!visible) filteredOutNodes.add(n.id);
            nodeUpdates.push({ id: n.id, hidden: !visible, opacity: 1.0 });
        });
        nodes.update(nodeUpdates);

        edges.get().forEach(function(e) {
            edgeUpdates.push({
                id: e.id,
                hidden: filteredOutNodes.has(e.from) || filteredOutNodes.has(e.to)
            });
        });
        edges.update(edgeUpdates);
        // keep the 3D view in sync if it has been opened
        if (typeof applyFilters3D === "function") applyFilters3D();
    }

    function toggleTopic(topic) {
        hiddenTopics.has(topic) ? hiddenTopics.delete(topic) : hiddenTopics.add(topic);
        var btn = document.querySelector('[data-topic="' + topic + '"]');
        if (btn) {
            var on = !hiddenTopics.has(topic);
            btn.style.opacity = on ? "1.0" : "0.3";
            btn.querySelector(".topic-label").style.textDecoration = on ? "none" : "line-through";
        }
        applyFilters();
    }

    function updateDegreeFilter(val) {
        minCitations = parseInt(val, 10);
        document.getElementById("degree-val").textContent = val;
        applyFilters();
    }

    function resetAllFilters() {
        hiddenTopics.clear();
        minCitations = 0;
        document.getElementById("degree-slider").value = "0";
        document.getElementById("degree-val").textContent = "0";
        document.querySelectorAll(".topic-filter").forEach(function(b) {
            b.style.opacity = "1.0";
            b.querySelector(".topic-label").style.textDecoration = "none";
        });
        applyFilters();
    }

    // clicking the legend title hides or shows every topic at once
    var allTopicKeys = Object.keys(
        nodes.get().reduce(function(acc, n) { if (n.topic) acc[n.topic] = 1; return acc; }, {})
    );

    function toggleAllTopics() {
        var allHidden = allTopicKeys.every(function(t) { return hiddenTopics.has(t); });
        var icon = document.getElementById("legend-toggle-icon");
        var title = document.getElementById("legend-title");

        if (allHidden) {
            // bring everything back
            hiddenTopics.clear();
            document.querySelectorAll(".topic-filter").forEach(function(b) {
                b.style.opacity = "1.0";
                b.querySelector(".topic-label").style.textDecoration = "none";
            });
            if (icon)  { icon.textContent = "\u25cf all"; icon.style.color = "#8b949e"; }
            if (title) { title.style.color = "#58a6ff"; }
        } else {
            // hide everything
            allTopicKeys.forEach(function(t) { hiddenTopics.add(t); });
            document.querySelectorAll(".topic-filter").forEach(function(b) {
                b.style.opacity = "0.3";
                b.querySelector(".topic-label").style.textDecoration = "line-through";
            });
            if (icon)  { icon.textContent = "\u25cb off"; icon.style.color = "#58a6ff"; }
            if (title) { title.style.color = "#8b949e"; }
        }
        applyFilters();
    }

    // click a node to dim everything except its direct neighbors
    network.on("click", function(params) {
        var allNodeObjs = nodes.get({ returnType: "Object" });
        var allEdgeObjs = edges.get({ returnType: "Object" });

        if (params.nodes.length > 0) {
            isHighlighted = true;
            var clickedId  = params.nodes[0];
            var neighbours = new Set(network.getConnectedNodes(clickedId));
            var connEdges  = new Set(network.getConnectedEdges(clickedId));

            var nodeUpdates = [];
            for (var nid in allNodeObjs) {
                if (filteredOutNodes.has(nid)) continue;
                var keep = (nid === clickedId || neighbours.has(nid));
                nodeUpdates.push({ id: nid, opacity: keep ? 1.0 : DIM_OPACITY });
            }
            nodes.update(nodeUpdates);

            var edgeUpdates = [];
            for (var eid in allEdgeObjs) {
                if (allEdgeObjs[eid].hidden) continue;
                var active = connEdges.has(eid) || connEdges.has(parseInt(eid, 10));
                edgeUpdates.push({ id: eid, color: active ? ACTIVE_EDGE : DIMMED_EDGE });
            }
            edges.update(edgeUpdates);

        } else if (isHighlighted) {
            isHighlighted = false;
            var nodeReset = [];
            for (var nid in allNodeObjs) {
                if (!filteredOutNodes.has(nid))
                    nodeReset.push({ id: nid, opacity: 1.0 });
            }
            nodes.update(nodeReset);

            var edgeReset = [];
            for (var eid in allEdgeObjs) {
                if (!allEdgeObjs[eid].hidden)
                    edgeReset.push({ id: eid, color: originalEdgeColors[eid] });
            }
            edges.update(edgeReset);
        }
    });

    </script>
    """

    # 3D view — uses 3d-force-graph (Three.js based) loaded from CDN
    three_d_js = """
    <script src="https://unpkg.com/3d-force-graph@1/dist/3d-force-graph.min.js"></script>
    <script type="text/javascript">

    var graph3D     = null;
    var currentView = "2d";
    var nodes3D = [];
    var links3D = [];

    // ego-network highlight state for 3D
    var hl3DNodes = new Set();
    var hl3DLinks = new Set();
    var is3DHl    = false;


    function linkIds(l) {
        var s = (l.source && l.source.id !== undefined) ? l.source.id : l.source;
        var t = (l.target && l.target.id !== undefined) ? l.target.id : l.target;
        return { s: s, t: t };
    }

    function nodeColor3D(n) {
        if (is3DHl) return hl3DNodes.has(n.id) ? n.color : n.color + "18";
        return n.color;
    }

    function linkColor3D(l) {
        var ids = linkIds(l);
        if (is3DHl) {
            var k1 = ids.s + ":::" + ids.t, k2 = ids.t + ":::" + ids.s;
            return (hl3DLinks.has(k1) || hl3DLinks.has(k2)) ? "#e0e0e0" : "#ffffff0a";
        }
        // same-topic edges get a tinted color; cross-topic ones stay grey
        var sN = nodes3D.find(function(x){ return x.id === ids.s; });
        var tN = nodes3D.find(function(x){ return x.id === ids.t; });
        if (sN && tN && sN.topic === tN.topic && sN.topic !== "Unknown")
            return sN.color + "cc";   // 80 % opacity, clearly visible
        return "#586069";             // medium grey instead of near-black
    }

    function linkWidth3D(l) {
        var ids = linkIds(l);
        if (is3DHl) {
            var k1 = ids.s + ":::" + ids.t, k2 = ids.t + ":::" + ids.s;
            return (hl3DLinks.has(k1) || hl3DLinks.has(k2)) ? 3.0 : 0.3;
        }
        return 1.5;
    }

    function redraw3D() {
        if (!graph3D) return;
        graph3D.nodeColor(nodeColor3D)
               .linkColor(linkColor3D)
               .linkWidth(linkWidth3D);
    }

    // called from applyFilters() to keep 3D in sync with the legend filters
    function applyFilters3D() {
        if (!graph3D) return;
        graph3D
            .nodeVisibility(function(n) {
                return !hiddenTopics.has(n.topic) && (n.citations || 0) >= minCitations;
            })
            .linkVisibility(function(l) {
                var ids  = linkIds(l);
                var data = graph3D.graphData();
                var sN   = data.nodes.find(function(x){ return x.id === ids.s; });
                var tN   = data.nodes.find(function(x){ return x.id === ids.t; });
                if (!sN || !tN) return true;
                return !hiddenTopics.has(sN.topic) && !hiddenTopics.has(tN.topic)
                    && (sN.citations || 0) >= minCitations
                    && (tN.citations || 0) >= minCitations;
            });
    }

    // switch between 2D and 3D views
    function setViewMode(mode) {
        if (mode === currentView) return;
        currentView = mode;

        var pyvisEl = document.getElementById("mynetwork");
        var el3d    = document.getElementById("graph-3d-container");
        var btn2d   = document.getElementById("btn-2d");
        var btn3d   = document.getElementById("btn-3d");

        // legend and stats are z-index 9999 so they always sit above the 3D canvas

        if (mode === "3d") {
            if (pyvisEl) pyvisEl.style.display = "none";
            if (el3d)    el3d.style.display    = "block";
            btn2d.style.background = "#161b22"; btn2d.style.color = "#8b949e";
            btn3d.style.background = "#58a6ff"; btn3d.style.color = "#0d1117";
            if (!graph3D) init3DGraph();
        } else {
            if (pyvisEl) pyvisEl.style.display = "";
            if (el3d)    el3d.style.display    = "none";
            btn2d.style.background = "#58a6ff"; btn2d.style.color = "#0d1117";
            btn3d.style.background = "#161b22"; btn3d.style.color = "#8b949e";
        }
    }

    // build the 3D graph on first switch to 3D mode
    function init3DGraph() {
        nodes3D = nodes.get().map(function(n) {
            return {
                id:        n.id,
                color:     (n.color && n.color.background) ? n.color.background : "#58a6ff",
                topic:     n.topic  || "Unknown",
                citations: n.citations || 0,
                val:       Math.max(0.5, ((n.size || 8) - 6) / 5)
            };
        });

        links3D = edges.get().map(function(e) {
            return { source: e.from, target: e.to };
        });

        var container = document.getElementById("graph-3d-container");

        graph3D = ForceGraph3D()(container)
            .width(container.clientWidth  || window.innerWidth)
            .height(container.clientHeight || window.innerHeight)
            .backgroundColor("#0d1117")

            .nodeColor(nodeColor3D)
            .nodeVal(function(n) { return n.val; })
            .nodeOpacity(0.95)
            .nodeResolution(14)
            .nodeLabel("")   // disable built-in label; we use onNodeHover instead

            .linkColor(linkColor3D)
            .linkWidth(linkWidth3D)
            .linkOpacity(0.85)
            .linkDirectionalParticles(0)

            .d3AlphaDecay(0.015)
            .d3VelocityDecay(0.25)
            .warmupTicks(0)
            .cooldownTime(8000)

            // reuse the same tooltip div as 2D
            .onNodeHover(function(node) {
                if (!node) { tooltipEl.style.display = "none"; return; }
                var topic = (node.topic || "Unknown").replace(/_/g, " ");
                var deg   = node.citations != null ? node.citations : "–";
                var col   = node.color;
                tooltipEl.style.border = "1px solid " + col;
                tooltipEl.innerHTML =
                    "<div style='color:" + col + ";font-weight:700;font-size:13px;margin-bottom:6px;'>"
                    + "&#9679;&nbsp;" + topic + "</div>"
                    + "<hr style='border:none;border-top:1px solid #30363d;margin:0 0 7px'/>"
                    + "<div style='color:#8b949e;display:grid;grid-template-columns:90px 1fr;row-gap:3px;'>"
                    + "<span>Paper ID</span><span style='color:#c9d1d9;'>"   + node.id  + "</span>"
                    + "<span>Citations</span><span style='color:#c9d1d9;'>"  + deg      + "</span>"
                    + "<span>Topic</span><span style='color:#c9d1d9;'>"      + topic    + "</span>"
                    + "</div>";
                tooltipEl.style.display = "block";
                placeTooltip();
            })

            .onNodeClick(function(node) {
                // second click on the same node resets the highlight
                if (is3DHl && hl3DNodes.size === 1 && hl3DNodes.has(node.id)) {
                    is3DHl = false; hl3DNodes.clear(); hl3DLinks.clear();
                } else {
                    is3DHl = true; hl3DNodes.clear(); hl3DLinks.clear();
                    hl3DNodes.add(node.id);
                    graph3D.graphData().links.forEach(function(l) {
                        var ids = linkIds(l);
                        if (ids.s === node.id) {
                            hl3DNodes.add(ids.t);
                            hl3DLinks.add(ids.s + ":::" + ids.t);
                        } else if (ids.t === node.id) {
                            hl3DNodes.add(ids.s);
                            hl3DLinks.add(ids.t + ":::" + ids.s);
                        }
                    });
                    // fly toward the clicked node
                    var dist  = 120;
                    var ratio = 1 + dist / Math.hypot(node.x || 1, node.y || 1, node.z || 1);
                    graph3D.cameraPosition(
                        { x: node.x * ratio, y: node.y * ratio, z: node.z * ratio },
                        node, 1200
                    );
                }
                redraw3D();
            })

            .onBackgroundClick(function() {
                if (is3DHl) { is3DHl = false; hl3DNodes.clear(); hl3DLinks.clear(); redraw3D(); }
            })

            .graphData({ nodes: nodes3D, links: links3D });

        // tune forces after graphData() so the force objects exist
        var chargeFn = graph3D.d3Force("charge");
        if (chargeFn) chargeFn.strength(-180).distanceMax(400);

        var linkFn = graph3D.d3Force("link");
        if (linkFn) linkFn.distance(80).strength(0.4).iterations(2);

        // resize the 3D canvas when the window changes size
        window.addEventListener("resize", function() {
            if (graph3D && currentView === "3d") {
                var c = document.getElementById("graph-3d-container");
                graph3D.width(c.clientWidth).height(c.clientHeight);
            }
        });
    }

    </script>
    """

    html = html.replace(
        "</body>",
        legend_div
        + stats_panel
        + loading_overlay
        + view_toggle_html
        + custom_tooltip
        + main_js
        + three_d_js
        + "\n</body>"
    )

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    print("  Done injecting UI and interactions.")


if __name__ == "__main__":
    print("\nCora Citation Network - Graph Visualization")
    print("-" * 45)

    print("\n[1/4] Downloading dataset...")
    content_path, cites_path = download_cora()

    print("[2/4] Building graph...")
    G = build_graph(content_path, cites_path)

    print("[3/4] Building visualization...")
    net = build_pyvis(G)
    net.save_graph(OUTPUT_HTML)

    print("[4/4] Injecting UI elements...")
    inject_interactions_and_legend(OUTPUT_HTML, G)

    print(f"\nDone! Open in browser: {os.path.abspath(OUTPUT_HTML)}\n")
