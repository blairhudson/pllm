-- Preserve compact two-column tables and numbered equations in the PDF.

local function has_class(element, class)
  for _, value in ipairs(element.classes) do
    if value == class then
      return true
    end
  end
  return false
end

local function write_latex(blocks)
  return (pandoc.write(pandoc.Pandoc(blocks), "latex"):gsub("%s+$", ""))
end

local function replace_plain(text, old, new)
  local pattern = old:gsub("(%W)", "%%%1")
  return (text:gsub(pattern, new))
end

local function web_math(text)
  if text:find("u%-Wr", 1, false) then
    return "u-Wr = W(x+r)-Wr = Wx mod p"
  end
  if text:find("Q_j", 1, true) then
    return "Q_j = sum_i W_j,i Enc(P_i)"
  end
  if text:find("P_i", 1, true) then
    return "P_i(X) = sum(b = 0 to B - 1) r_b,i X^b"
  end
  if text:find("C=WA", 1, true) then
    return "C = WA mod q"
  end
  if text:find("L_{\\rm exchange}", 1, true) then
    return "T_k = (L_exchange + (k + 1) C_row + C_replay) / A_k"
  end
  if text:find("W\\in\\mathbb", 1, true) then
    return "W in Z^(m x n)"
  end

  local replacements = {
    { "\\operatorname{Enc}", "Enc" },
    { "\\mathbb Z", "Z" },
    { "\\mathbb{Z}", "Z" },
    { "\\pmod p", " mod p" },
    { "\\pmod q", " mod q" },
    { "\\times", " x " },
    { "\\in", " in " },
    { "\\leq", " <= " },
    { "\\pm", "+/-" },
    { "\\rm ", "" },
  }
  for _, replacement in ipairs(replacements) do
    text = replace_plain(text, replacement[1], replacement[2])
  end
  return (text:gsub("[{}]", ""):gsub("\\", ""))
end

local function table_latex(div)
  local table_element
  local caption
  for _, block in ipairs(div.content) do
    if block.t == "Table" then
      table_element = block
    elseif block.t == "Para" then
      caption = block
    end
  end
  if table_element == nil or caption == nil then
    error("paper-table div requires one table and a caption paragraph")
  end

  local rows = {}
  local function append_row(row)
    local cells = {}
    for _, cell in ipairs(row.cells) do
      table.insert(cells, write_latex(cell.contents))
    end
    table.insert(rows, table.concat(cells, " & ") .. " \\\\")
  end

  for _, row in ipairs(table_element.head.rows) do
    append_row(row)
  end
  table.insert(rows, "\\midrule")
  for _, body in ipairs(table_element.bodies) do
    for _, row in ipairs(body.body) do
      append_row(row)
    end
  end

  local columns = div.attributes["latex-columns"] or string.rep("l", #table_element.colspecs)
  local environment = "tabular"
  local arguments = "{" .. columns .. "}"
  if columns:find("X", 1, true) then
    environment = "tabularx"
    arguments = "{\\columnwidth}{" .. columns .. "}"
  end
  local latex = {
    "\\begin{table}[t]",
    "\\centering\\small",
    "\\setlength{\\tabcolsep}{4.5pt}",
    "\\begin{" .. environment .. "}" .. arguments,
    "\\toprule",
    table.remove(rows, 1),
    table.concat(rows, "\n"),
    "\\bottomrule",
    "\\end{" .. environment .. "}",
    "\\caption{" .. write_latex({ caption }) .. "}",
    "\\label{" .. div.identifier .. "}",
    "\\end{table}",
  }
  return pandoc.RawBlock("latex", table.concat(latex, "\n"))
end

function Div(div)
  if has_class(div, "web-only") then
    if FORMAT:match("latex") then
      return {}
    end
    return div.content
  end
  if has_class(div, "paper-table") then
    if FORMAT:match("latex") then
      return table_latex(div)
    end
    return div.content
  end
  if has_class(div, "numbered-equation") then
    local equation
    div:walk({
      Math = function(math)
        if math.mathtype == "DisplayMath" then
          equation = (math.text:gsub("^%s+", ""):gsub("%s+$", ""))
        end
      end,
    })
    if equation == nil then
      error("numbered-equation div requires display math")
    end
    if FORMAT:match("latex") then
      return pandoc.RawBlock("latex", "\\begin{equation}\n" .. equation .. "\n\\end{equation}")
    end
    return pandoc.CodeBlock(web_math(equation), { class = "text" })
  end
  return div
end

function Math(math)
  if not FORMAT:match("latex") and math.mathtype == "InlineMath" then
    return pandoc.Code(web_math(math.text))
  end
  return math
end
