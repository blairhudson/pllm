function Math(element)
  return pandoc.Code(element.text)
end

function Div(element)
  return element.content
end

function Link(element)
  if pandoc.utils.stringify(element.content) == element.target then
    element.content = {pandoc.Str(element.target:gsub("^https?://", ""))}
  end
  return element
end
