---
title: "$title$"
description: "$description$"
---

<div className="paper-actions">
  <a className="paper-action paper-action-primary" href="/downloads/$pdf$">Download PDF <span aria-hidden="true">↗</span></a>
$if(source)$
  <a className="paper-action" href="/downloads/$source$">Markdown source</a>
$endif$
$if(arxiv)$
  <a className="paper-action" href="/downloads/$arxiv$">arXiv TeX source</a>
$endif$
</div>

To run or extend an experiment, follow the [research workflow](/research/recipes/experiments/).

$if(author)$
**$for(author)$$author$$sep$, $endfor$**$if(affiliation)$ · $affiliation$$endif$$if(date)$ · $date$$endif$
$endif$

$if(abstract)$
## Abstract

$abstract$
$endif$

$body$
