local function latex(blocks)
  local value = pandoc.write(pandoc.Pandoc(blocks), "latex"):gsub("%s+$", ""):gsub("\n", " ")
  return value
end

local function rows(element)
  local result = {}
  for _, row in ipairs(element.head.rows) do table.insert(result, row) end
  for _, body in ipairs(element.bodies) do
    for _, row in ipairs(body.head) do table.insert(result, row) end
    for _, row in ipairs(body.body) do table.insert(result, row) end
  end
  for _, row in ipairs(element.foot.rows) do table.insert(result, row) end
  return result
end

function Table(element)
  if not FORMAT:match("latex") then return nil end

  local all_rows = rows(element)
  local spanning = #element.colspecs > 3

  local columns = {}
  for _, spec in ipairs(element.colspecs) do
    local alignment = tostring(spec[1])
    table.insert(columns, alignment == "AlignRight" and "r" or alignment == "AlignCenter" and "c" or ">{\\raggedright\\arraybackslash}X")
  end

  local output = {
    spanning and "\\begin{table*}[t]" or "\\par\\medskip\\noindent\\begin{minipage}{\\linewidth}",
    "\\centering\\small",
  }
  if #element.caption.long > 0 then
    table.insert(output, spanning and "\\caption{" .. latex(element.caption.long) .. "}" or "\\textbf{" .. latex(element.caption.long) .. "}\\par\\smallskip")
  end
  table.insert(output, "\\begin{tabularx}{\\linewidth}{@{}" .. table.concat(columns) .. "@{}}")
  table.insert(output, "\\toprule")
  for index, row in ipairs(all_rows) do
    local cells = {}
    for _, cell in ipairs(row.cells) do table.insert(cells, latex(cell.contents)) end
    table.insert(output, table.concat(cells, " & ") .. " \\\\")
    if index == #element.head.rows then table.insert(output, "\\midrule") end
  end
  table.insert(output, "\\bottomrule")
  table.insert(output, "\\end{tabularx}")
  table.insert(output, spanning and "\\end{table*}" or "\\end{minipage}\\par\\medskip")
  return pandoc.RawBlock("latex", table.concat(output, "\n"))
end

function Code(element)
  if FORMAT:match("latex") and element.text:match("^[%w_.%-]+%.json$") then
    return pandoc.RawInline("latex", "\\nolinkurl{" .. element.text .. "}")
  end
end

function Figure(element)
  if not FORMAT:match("latex") then return nil end
  local block = element.content[1]
  local image = block and block.content and block.content[1]
  if not image or image.t ~= "Image" or not image.src:match("^figures/[%w%-]+%.png$") then
    return nil
  end
  local caption = latex(element.caption.long)
  return pandoc.RawBlock("latex", table.concat({
    "\\par\\smallskip\\noindent\\begin{minipage}{\\linewidth}\\centering",
    "\\includegraphics[width=0.89\\linewidth]{docs/build/papers/" .. image.src .. "}",
    "\\par\\smallskip\\footnotesize\\emph{" .. caption .. "}",
    "\\end{minipage}\\par\\smallskip",
  }, "\n"))
end
