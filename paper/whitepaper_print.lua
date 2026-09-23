-- Split the canonical Markdown into intentional print spreads. No copy lives here.
local function text(value)
  return pandoc.utils.stringify(value)
end

local function escape(value)
  return text(value):gsub("&", "&amp;"):gsub("<", "&lt;"):
    gsub(">", "&gt;"):gsub('"', "&quot;")
end

function Pandoc(doc)
  if not FORMAT:match("html") then return nil end

  local sections = {{}, {}, {}}
  local current = 1
  local seen = {}
  local breaks = {
    ["Research becomes a candidate capability"] = 2,
    ["Measured baseline; open comparison"] = 3,
  }
  for _, block in ipairs(doc.blocks) do
    if block.t == "Header" and block.level == 2 then
      local heading = text(block.content)
      seen[heading] = true
      current = breaks[heading] or current
    end
    table.insert(sections[current], block)
  end
  for heading in pairs(breaks) do
    if not seen[heading] then error("Missing whitepaper print section: " .. heading) end
  end
  if not seen["Why PLLM"] or not seen["One private request"] or
     not seen["Economic value needs an honest denominator"] then
    error("Whitepaper print sections changed; review page composition")
  end

  local title = text(doc.meta.title)
  local subject = title:match("^PLLM:%s*(.+)$")
  if not subject then error("Whitepaper print title must begin PLLM:") end
  local hero = table.concat({
    '<header class="cover-hero">',
    '<div class="cover-top"><span class="wordmark">PLLM<span class="wordmark-dot">.</span></span>',
    '<span class="edition">WHITEPAPER / EDITION ' .. escape(doc.meta.edition) .. '</span></div>',
    '<div class="cover-rule"></div>',
    '<h1>' .. escape(subject) .. '</h1>',
    '<div class="cover-bottom"><span>' .. escape(doc.meta.author) .. '</span>',
    '<span>' .. escape(doc.meta["web-date"]) .. '</span></div>',
    '</header>',
  })
  table.insert(sections[1], 1, pandoc.RawBlock("html", hero))

  local labels = {
    "PRIVATE INFERENCE / TWO ROLES ONLINE",
    "RESEARCH / COMPOSITION & VALUE",
    "EVIDENCE / WHAT WE HAVE MEASURED",
  }
  local pages = {}
  for index, blocks in ipairs(sections) do
    if index > 1 then
      table.insert(blocks, 1, pandoc.RawBlock("html", '<div class="page-top">' .. labels[index] .. '</div>'))
    end
    table.insert(pages, pandoc.Div(blocks, pandoc.Attr("", {"sheet", "sheet-" .. index}, {
      ["data-page"] = string.format("%02d", index),
    })))
  end
  return pandoc.Pandoc(pages, doc.meta)
end
