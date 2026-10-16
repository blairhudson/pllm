local function latex(blocks)
  local value = pandoc.write(pandoc.Pandoc(blocks), "latex"):gsub("%s+$", ""):gsub("\n", " ")
  return value
end

function Meta(metadata)
  if FORMAT:match("latex") and metadata.affiliation and metadata.email then
    local author = pandoc.utils.stringify(metadata.author)
    local affiliation = pandoc.utils.stringify(metadata.affiliation)
    local email = pandoc.utils.stringify(metadata.email)
    metadata.author = pandoc.MetaList({pandoc.MetaInlines({pandoc.RawInline("latex",
      author .. "\\\\{\\small " .. affiliation .. "}\\\\{\\small\\texttt{" .. email .. "}}")})})
    return metadata
  end
end

function Str(element)
  if FORMAT:match("latex") then
    -- The manuscript uses six-per-em spaces between values and units; Latin
    -- Modern does not contain that Unicode glyph in either supported engine.
    element.text = element.text:gsub(" ", " ")
    return element
  end
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
  local manual_caption = element.attributes["pllm-manual-caption"] == "true"
  if #element.caption.long > 0 and not manual_caption then
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
  if manual_caption then
    table.insert(output, "\\par\\smallskip\\raggedright " .. latex(element.caption.long))
  end
  table.insert(output, spanning and "\\end{table*}" or "\\end{minipage}\\par\\medskip")
  return pandoc.RawBlock("latex", table.concat(output, "\n"))
end

local function attach_table_captions(blocks)
  if not FORMAT:match("latex") then return nil end
  local result = pandoc.List()
  local index = 1
  while index <= #blocks do
    local block, following = blocks[index], blocks[index + 1]
    if block.t == "Table" and #block.caption.long == 0 and following
        and following.t == "Para" and following.content[1]
        and following.content[1].t == "Strong"
        and pandoc.utils.stringify(following.content[1]):match("^Table %d+%.$") then
      -- Keep the author-numbered Markdown caption inside its floating table.
      -- Otherwise a wide table can move to the next page without its scope.
      block.caption.long = { following }
      block.attributes["pllm-manual-caption"] = "true"
      index = index + 1
    end
    result:insert(block)
    index = index + 1
  end
  return result
end

function Code(element)
  if not FORMAT:match("latex") then return nil end
  local digest = element.text:match("^%x+$")
    and (#element.text == 40 or #element.text == 64)
  if element.text:match("^[%w_.%-]+%.json$") or digest then
    return pandoc.RawInline("latex", "\\nolinkurl{" .. element.text .. "}")
  end
  -- Preserve simple commands, with breaks at spaces and within path tokens.
  -- A long path can otherwise cross the gutter even with breakable spaces.
  if element.text:match("^[%w_./%-]+ [%w_./ %-]+$")
      and not element.text:find("  ", 1, true) and not element.text:match(" $") then
    local words = {}
    for word in element.text:gmatch("%S+") do
      local rendered = word:find("/", 1, true)
        and "\\texttt{\\nolinkurl{" .. word .. "}}"
        or latex({ pandoc.Plain({ pandoc.Code(word) }) })
      table.insert(words, rendered)
    end
    return pandoc.RawInline("latex", table.concat(words, " "))
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

return {
  { Blocks = attach_table_captions },
  { Meta = Meta, Str = Str, Table = Table, Code = Code, Figure = Figure },
}
