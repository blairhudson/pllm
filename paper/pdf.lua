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
  local wide = false
  for _, row in ipairs(all_rows) do
    for _, cell in ipairs(row.cells) do
      if #pandoc.utils.stringify(cell.contents) > 32 then wide = true end
    end
  end

  local columns = {}
  for _, spec in ipairs(element.colspecs) do
    local alignment = tostring(spec[1])
    table.insert(columns, alignment == "AlignRight" and "r" or alignment == "AlignCenter" and "c" or "l")
  end
  if wide then
    columns = {}
    for _ = 1, #element.colspecs do table.insert(columns, "X") end
  end

  local output = {
    "\\begin{table*}[t]",
    "\\centering\\small",
  }
  if #element.caption.long > 0 then
    table.insert(output, "\\caption{" .. latex(element.caption.long) .. "}")
  end
  table.insert(output, wide and "\\begin{tabularx}{\\textwidth}{@{}" .. table.concat(columns) .. "@{}}" or "\\begin{tabular}{@{}" .. table.concat(columns) .. "@{}}")
  table.insert(output, "\\toprule")
  for index, row in ipairs(all_rows) do
    local cells = {}
    for _, cell in ipairs(row.cells) do table.insert(cells, latex(cell.contents)) end
    table.insert(output, table.concat(cells, " & ") .. " \\\\")
    if index == #element.head.rows then table.insert(output, "\\midrule") end
  end
  table.insert(output, "\\bottomrule")
  table.insert(output, wide and "\\end{tabularx}" or "\\end{tabular}")
  table.insert(output, "\\end{table*}")
  return pandoc.RawBlock("latex", table.concat(output, "\n"))
end
