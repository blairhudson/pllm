function Math(element)
  return pandoc.Code((element.text:gsub("\n", " ")))
end

function CodeBlock(element)
  -- MDX does not recognize indented code: keep diagrams fenced as text.
  if #element.classes == 0 then element.classes:insert("text") end
  return element
end

function Div(element)
  return element.content
end

function Link(element)
  -- GFM cannot preserve Pandoc's autolink class in Markdown. Its raw HTML
  -- fallback lets MDX auto-link a www label again, creating nested anchors.
  element.classes = element.classes:filter(function(value) return value ~= "uri" end)
  if pandoc.utils.stringify(element.content) == element.target then
    element.content = {pandoc.Str(element.target:gsub("^https?://", ""))}
  end
  return element
end

function Image(element)
  local figure = element.src:match("^figures/([%w%-]+%.png)$")
  if figure then element.src = "/downloads/figures/" .. figure end
  return element
end
