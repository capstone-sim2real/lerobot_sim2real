import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { readFileSync, writeFileSync } from 'node:fs';
import { load } from 'cheerio';
import { Button } from './components/ui/button';
import { Card } from './components/ui/card';
import { Badge } from './components/ui/badge';
import { Input } from './components/ui/input';

// Static shadcn rendering keeps existing native control IDs and event ownership.
// No hydration, runtime CDN, or second controller state is introduced.
const $=load(readFileSync('index.template.html','utf8'));
function decorate(selector: string, render: (className: string, el: ReturnType<typeof $>)=>string) {
  $(selector).each((_,node)=>{
    const el=$(node);
    const rendered=load(render(el.attr('class')??'',el),null,false).root().children().first();
    const attrs=rendered.attr()??{};
    for(const [name,value] of Object.entries(attrs))el.attr(name,value);
  });
}
decorate('button', (className,el)=>renderToStaticMarkup(<Button className={className}
  variant={el.attr('id')==='stop'?'destructive':el.hasClass('btn-primary')?'default':el.hasClass('link')?'ghost':'outline'}/>));
decorate('.panel',className=>renderToStaticMarkup(<Card className={className}/>));
decorate('.badge',className=>renderToStaticMarkup(<Badge className={className} variant="secondary"/>));
decorate('input:not([type=checkbox]):not([type=radio])',className=>renderToStaticMarkup(<Input className={className}/>));
$('head').append('<link rel="stylesheet" href="/shadcn.css?v=__SHADCN_CSS_VERSION__">');
writeFileSync('../src/agent/web/index.html',$.html());
