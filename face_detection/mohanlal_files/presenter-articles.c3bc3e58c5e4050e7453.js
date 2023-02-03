!function(e){function t(t){for(var n,r,l=t[0],s=t[1],d=t[2],p=0,g=[];p<l.length;p++)r=l[p],Object.prototype.hasOwnProperty.call(o,r)&&o[r]&&g.push(o[r][0]),o[r]=0;for(n in s)Object.prototype.hasOwnProperty.call(s,n)&&(e[n]=s[n]);for(c&&c(t);g.length;)g.shift()();return a.push.apply(a,d||[]),i()}function i(){for(var e,t=0;t<a.length;t++){for(var i=a[t],n=!0,l=1;l<i.length;l++){var s=i[l];0!==o[s]&&(n=!1)}n&&(a.splice(t--,1),e=r(r.s=i[0]))}return e}var n={},o={6:0},a=[];function r(t){if(n[t])return n[t].exports;var i=n[t]={i:t,l:!1,exports:{}};return e[t].call(i.exports,i,i.exports,r),i.l=!0,i.exports}r.m=e,r.c=n,r.d=function(e,t,i){r.o(e,t)||Object.defineProperty(e,t,{enumerable:!0,get:i})},r.r=function(e){"undefined"!=typeof Symbol&&Symbol.toStringTag&&Object.defineProperty(e,Symbol.toStringTag,{value:"Module"}),Object.defineProperty(e,"__esModule",{value:!0})},r.t=function(e,t){if(1&t&&(e=r(e)),8&t)return e;if(4&t&&"object"==typeof e&&e&&e.__esModule)return e;var i=Object.create(null);if(r.r(i),Object.defineProperty(i,"default",{enumerable:!0,value:e}),2&t&&"string"!=typeof e)for(var n in e)r.d(i,n,function(t){return e[t]}.bind(null,n));return i},r.n=function(e){var t=e&&e.__esModule?function(){return e.default}:function(){return e};return r.d(t,"a",t),t},r.o=function(e,t){return Object.prototype.hasOwnProperty.call(e,t)},r.p="/verso/static/";var l=window.webpackJsonpVerso=window.webpackJsonpVerso||[],s=l.push.bind(l);l.push=t,l=l.slice();for(var d=0;d<l.length;d++)t(l[d]);var c=s;a.push([2653,0,1]),i()}({129:function(e,t,i){const{default:n,css:o}=i(3),{calculateSpacing:a,getColorStyles:r,getTypographyStyles:l}=i(4),{BaseText:s}=i(10),{BREAKPOINTS:d}=i(6),c=i(11),p=n.header.withConfig({displayName:"UtilityLedeHeader"})`
  ${({contentAlign:e})=>e?(e=>`\n  text-align: ${e};\n`)(e):""}
  ${({theme:e,hasBackground:t})=>t?(e=>`\n  ${r(e,"background-color","colors.discovery.body.white.background")};\n`)(e):""}
  ${({hasImage:e})=>e?`\n      display: grid;\n      grid-template-columns: repeat(8, 1fr);\n      grid-column-gap: 1.5rem;\n      align-items: center;\n      padding: ${a(12)} 0 ${a(100)} 0;\n      @media  (min-width: 0) and (max-width: ${d.md}){\n          grid-gap: ${a(2)};\n          grid-template-columns: repeat(4, 1fr);\n          padding: ${a(5)} 0 ${a(5)} 0;\n          justify-items: center;\n      }\n      `:""}
  ${({alternativeStyle:e})=>e?`\n    grid-gap: ${a(2)} 0;\n    @media (min-width: 0) and (max-width: ${d.md}){\n      grid-gap: ${a(4)} 0;\n    }\n    `:""}
`,g=n.div.withConfig({displayName:"UtilityLedeWrapper"})`
  grid-auto-flow: row;
  grid-column: 4 / span 5;

  @media (min-width: 0) and (max-width: ${d.md}) {
    grid-gap: ${a(2)} 0;
    grid-column: 1 / span 4;
  }

  ${e=>!0===e.alternativeStyle?`\n        grid-gap: ${a(2)} 0;\n        @media (min-width: 0) and (max-width: ${d.md}){\n            grid-gap: ${a(4)} 0;\n        }\n        `:""}
`,u=n(s).withConfig({displayName:"UtilityLedeHedText"})`
  ${e=>!0===e.alternativeStyle?`\n        ${({theme:e})=>r(e,"color","colors.discovery.lead.primary.hed")};\n        ${({theme:e})=>l(e,"typography.definitions.discovery.page-hed-section")};\n        `:""}
  ${e=>!0===e.hasImage?"\n        margin: 0;\n        ":`margin: 0 0 ${a(2,"px")};`}

  ${e=>!0===e.hasInverted&&o`
      ${({theme:e})=>r(e,"color","colors.discovery.body.white.context-texture")};
      ${({theme:e})=>l(e,"typography.definitions.discovery.hed-core-primary")};
      line-height: 25px;
      font-size: 20px;
    `}
`;u.defaultProps={as:"h1",colorToken:"colors.discovery.body.white.heading",typeIdentity:"typography.definitions.discovery.hed-break-out"};const m=n(s).withConfig({displayName:"UtilityLedeDekText"})`
  a {
    ${({theme:e})=>r(e,"color","colors.discovery.body.white.accent")};
  }

  p {
    margin: 0; /* overwrite native browser margins for paragraph tags */
  }
  ${e=>!0===e.alternativeStyle?`\n        ${({theme:e})=>r(e,"color","colors.discovery.lead.primary.description")};\n        ${({theme:e})=>l(e,"typography.definitions.discovery.description.feature")};\n        a {\n          ${({theme:e})=>r(e,"color","colors.discovery.lead.primary.description")};\n        }\n        `:""}

  ${e=>!0===e.hasImage?"\n        margin: 0;\n        ":`margin: ${a(2,"px")} 0 0;`}

  @media (min-width: 0) and (max-width: ${d.md}) {
    grid-row: 2;
  }
`;m.defaultProps={as:"h2",colorToken:"colors.discovery.body.white.description",typeIdentity:"typography.definitions.consumptionEditorial.body-core"};const h=n(c).withConfig({displayName:"UtilityLedeImage"})`
  grid-column: 1 / span 3;
  @media (min-width: 0) and (max-width: ${d.md}) {
    grid-column: 2 / span 2;
    grid-row: 1;
  }
  ${e=>!0===e.hasImage?`\n      @media (min-width: 0) and (max-width: ${d.md}){\n        padding-bottom: 0;\n      }\n      `:""}
`;e.exports={UtilityLedeHeader:p,UtilityLedeWrapper:g,UtilityLedeHedText:u,UtilityLedeDekText:m,UtilityLedeImage:h}},130:function(e,t,i){const n=i(3).default,{BaseText:o}=i(10),{calculateSpacing:a,getColorStyles:r}=i(4),{BREAKPOINTS:l}=i(6),s=n.div.withConfig({displayName:"ToggleChipListWrapper"})`
  ${({hasToggleGridColor:e,theme:t})=>e&&`\n  ${r(t,"background-color","colors.background.light")};\n  border-bottom:${a(4)} solid ;\n  ${r(t,"border-color","colors.background.light")};\n  `}
  display: flex;
  flex-direction: column;
  ${({contentAlign:e})=>"left"===e?"align-items: flex-start":"center"===e?"align-items: center":"right"===e&&"align-items: flex-end"}
`,d=n(o).withConfig({displayName:"LabelText"})`
  margin: 0 0 ${a(1)};

  ${({hasIncreasedBottomMargin:e})=>e&&` \n    margin: 0 0 ${a(2)};\n  `}

  ${({hasLargeLabel:e})=>e&&" \n    font-size: 1rem;\n    letter-spacing: 1px;\n  "}

  ${({fullPageTheme:e,theme:t})=>"inverted"===e?r(t,"color","colors.interactive.base.white"):""}
`;d.defaultProps={as:"div",colorToken:"colors.interactive.base.black",typeIdentity:"typography.definitions.utility.label"};const c=n.div.withConfig({displayName:"ListWrapper"})`
  display: flex;
  flex-direction: row;
  padding: 0 ${a(3)};
  width: 100%;
  gap: ${a(1.5)};

  ${({hasNoHorizontalScroll:e})=>e&&" &::-webkit-scrollbar \n    {\n      display: none;\n    }"}

  ${({layout:e})=>"wrap"===e?"flex-wrap: wrap;":"overflow-x: auto;"}
  
  ${({contentAlign:e,layout:t})=>{if("nowrap"===t)return"";switch(e){case"left":return"justify-content: flex-start;";case"center":return"justify-content: center;";case"right":return"justify-content: flex-end;";default:return""}}}
  
  ${({hasToggleGridColor:e})=>e&&`--grid-margin: ${a(3)};`}
  ${({isReadViewShopViewEnabled:e})=>e&&`\n        padding:${a(.5)};\n         gap: ${a(.5)};\n        `}
`,p=n.div.withConfig({displayName:"ListItemWrapper"})`
  ${({contentAlign:e,layout:t,hasSpacing:i})=>{if("wrap"===t)return"";let n=e;switch(i&&(n="centerWithPadding"),n){case"center":return"\n        &:first-child {\n          margin-left: auto;\n        }\n\n        &:last-child {\n          margin-right: auto;\n        }\n      ";case"right":return"\n        &:first-child {\n          margin-left: auto;\n        }\n      ";case"centerWithPadding":return`\n        @media (min-width: ${l.md}) {\n          &:first-child {\n            margin-left: auto;\n          }\n\n          &:last-child {\n            margin-right: auto;\n          }\n        }\n\n        @media (min-width: ${l.sm}) and (max-width: ${l.md}) {\n          &:first-child {\n            margin-left: 1.5rem;\n          }\n  \n          &:last-child {\n            margin-right: 1.5rem;\n          }\n        }          \n        `;default:return""}}}
`,g=n(o).withConfig({displayName:"HelperText"})`
  margin: ${a(1)} 0 0;
`;g.defaultProps={as:"div",colorToken:"colors.interactive.base.dark",typeIdentity:"typography.definitions.utility.assistive-text"},e.exports={ToggleChipListWrapper:s,LabelText:d,ListWrapper:c,ListItemWrapper:p,HelperText:g}},1422:function(e,t,i){const n=i(1),o=i(0),{InfoSliceValue:a,InfoSliceKey:r,InfoSliceItem:l,InfoSliceListItem:s,InfoSliceList:d,InfoSliceWrapper:c}=i(1423),{trackComponent:p}=i(5),g=({className:e,items:t,isMusicReview:i})=>(o.useEffect(()=>{p("InfoSlice")},[]),t&&0!==t.length?o.createElement(c,{className:e},o.createElement(d,{isMusicReview:i},t.map(e=>{const{key:t,value:i}=e;return t&&i?o.createElement(s,{key:t.toString().toLowerCase()},o.createElement(l,null,o.createElement(r,null,t),o.createElement(a,null,i))):null}))):null);g.propTypes={className:n.string,isMusicReview:n.bool,items:n.arrayOf(n.shape({key:n.string,value:n.string}))},g.defaultProps={isMusicReview:!1},e.exports=g},1423:function(e,t,i){const n=i(3).default,{calculateSpacing:o,getColorStyles:a}=i(4),{BaseText:r}=i(10),{BREAKPOINTS:l}=i(6),{maxThresholds:s}=i(17),d=n(r).withConfig({displayName:"InfoSliceValue"})`
  display: table-cell;
  vertical-align: top;
`;d.defaultProps={colorToken:"colors.consumption.body.standard.body",typeIdentity:"typography.definitions.globalEditorial.numerical-small"};const c=n(r).withConfig({displayName:"InfoSliceKey"})`
  display: table-cell;
  padding-right: ${o(1)};
  vertical-align: top;
`;c.defaultProps={colorToken:"colors.consumption.body.standard.subhed",typeIdentity:"typography.definitions.globalEditorial.context-title"};const p=n.div.withConfig({displayName:"InfoSliceItem"})`
  display: table;
  align-items: center;
  padding: ${o(.5)} 0;
`,g=n.li.withConfig({displayName:"InfoSliceListItem"})`
  @media (max-width: ${s.md}px) {
    margin: 0 auto;
  }
  @media (min-width: ${l.md}) {
    margin-right: ${o(3)};

    :last-child {
      margin-right: ${o(0)};
    }
  }
`,u=n.ul.withConfig({displayName:"InfoSliceList"})`
  display: flex;
  flex-direction: column;
  margin: 0;
  border-width: 2px 0 0;
  border-style: solid;
  ${({theme:e})=>a(e,"border-color","colors.consumption.body.standard.divider")};
  padding: ${o(1.5)} 0;
  list-style: none;

  @media (min-width: ${l.md}) {
    flex-direction: row;
    flex-wrap: wrap;
    align-items: center;
    ${({isMusicReview:e})=>e&&"\n    justify-content: center;\n    border-width: 0 0 0;"}
  }
  ${({isMusicReview:e})=>e&&"\n  justify-content: center;\n  border-width: 0 0 0;"}
`,m=n.div.withConfig({displayName:"InfoSliceWrapper"})``;e.exports={InfoSliceValue:d,InfoSliceKey:c,InfoSliceItem:p,InfoSliceListItem:g,InfoSliceList:u,InfoSliceWrapper:m}},1430:function(e,t,i){const{asVariation:n}=i(12),o=i(1441),a=i(1458),r=i(1461);o.TextAboveCenterGridWidth=n(o,"TextAboveCenterGridWidth",{contentAlign:"center",contentPosition:"above"}),o.TextAboveCenterGridWidthTopCardSmall=n(o,"TextAboveCenterGridWidthTopCardSmall",{contentAlign:"center",contentPosition:"above",contentStyle:"card",copyWidth:"fullbleed",leadRailAnchor:!0,mediaWidth:"small"}),o.TextAboveCenterFullBleed=n(o,"TextAboveCenterFullBleed",{contentAlign:"center",contentPosition:"above",mediaWidth:"fullbleed"}),o.TextAboveCenterFullBleedNoContributor=n(o,"TextAboveCenterFullBleedNoContributor",{contentAlign:"center",contentPosition:"above",mediaWidth:"fullbleed"},{hasDesktopTitleBlockDivider:!1,captionStyle:"span-content-well",captionWidth:"fullbleed",showContributorImage:!1}),o.TextAboveCenterFullBleedTop=n(o,"TextAboveCenterFullBleedTop",{contentAlign:"center",contentPosition:"above",copyWidth:"fullbleed"}),o.TextAboveCenterFullBleedGridWidthCard=n(o,"TextAboveCenterFullBleedGridWidthCard",{contentAlign:"center",contentPosition:"above",contentStyle:"card",copyWidth:"fullbleed"}),o.TextAboveCenterFullBleedCard=n(o,"TextAboveCenterFullBleedCard",{contentAlign:"center",contentPosition:"above",contentStyle:"card",mediaWidth:"fullbleed",copyWidth:"fullbleed"}),o.TextAboveLeftSmall=n(o,"TextAboveLeftSmall",{contentAlign:"left",contentPosition:"above",leadRailAnchor:!0,mediaWidth:"small"}),o.TextAboveLeftSmallWithRule=n(o,"TextAboveLeftSmallWithRule",{contentAlign:"left",contentPosition:"above",hasLedeLightbox:!0,leadRailAnchor:!0,mediaWidth:"smallrule",hasInlinePublishDate:!0,hasNoRowPadding:!0,hasXsNavSpacing:!0},{stickySocialAnchorTop:{selector:"[data-testid='ContentHeaderAccreditation']",edge:"bottom"}}),o.TextAboveCenterSmallWithRule=n(o,"TextAboveCenterSmallWithRule",{contentAlign:"center",contentPosition:"above",leadRailAnchor:!0,ledeAlign:"center",mediaWidth:"smallrule",hasInlinePublishDate:!0,hasNoRowPadding:!0,hasXsNavSpacing:!0},{stickySocialAnchorTop:{selector:"[data-testid='ContentHeaderAccreditation']",edge:"bottom"}}),o.InlineImage=n(o,"InlineImage",{contentAlign:"center",contentPosition:"above",leadRailAnchor:!0,ledeAlign:"center",mediaWidth:"smallrule",hasInlinePublishDate:!0,hasNoRowPadding:!0,hasXsNavSpacing:!0},{dividerType:"bottom",hideLede:!0,showContributorImage:!1,stickySocialAnchorTop:{selector:"[data-testid='ContentHeaderAccreditation']",edge:"bottom"}}),o.TextAboveLeftFullBleed=n(o,"TextAboveLeftFullBleed",{contentAlign:"left",contentPosition:"above",copyWidth:"grid",hasLedeLightbox:!0,mediaWidth:"fullbleed"}),o.TextAboveLeftGridWidth=n(o,"TextAboveLeftGridWidth",{contentAlign:"left",contentPosition:"above",mediaWidth:"grid"}),o.TextAboveLeftGridWidthCard=n(o,"TextAboveLeftGridWidthCard",{contentAlign:"left",contentPosition:"above",contentStyle:"card",mediaWidth:"grid"}),o.TextAboveLeftNoImg=n(o,"TextAboveLeftNoImg",{contentAlign:"left",contentPosition:"above"},{className:"content-header--no-lede",lede:null}),o.TextBelowCenterGridWidth=n(o,"TextBelowCenterGridWidth",{contentAlign:"center",contentPosition:"below"}),o.TextBelowCenterFullBleed=n(o,"TextBelowCenterFullBleed",{contentAlign:"center",contentPosition:"below",mediaWidth:"fullbleed"}),o.TextBelowCenterFullBleedNoContributor=n(o,"TextBelowCenterFullBleedNoContributor",{contentAlign:"center",contentPosition:"below",mediaWidth:"fullbleed"},{hasDesktopTitleBlockDivider:!1,captionStyle:"span-content-well",captionWidth:"fullbleed",showContributorImage:!1}),o.TextBelowLeftGridWidth=n(o,"TextBelowLeftGridWidth",{contentAlign:"left",contentPosition:"below"}),o.TextBelowLeftGridWidthCard=n(o,"TextBelowLeftGridWidthCard",{contentAlign:"left",contentPosition:"below",contentStyle:"card"}),o.TextBelowLeftFullBleed=n(o,"TextBelowLeftFullBleed",{contentAlign:"left",contentPosition:"below",mediaWidth:"fullbleed"}),o.TextBelowFullBleedDenseCard=n(o,"TextBelowFullBleedDenseCard",{contentAlign:"left",contentPosition:"below",contentStyle:"dense-card",mediaWidth:"fullbleed"}),o.TextOverlay=n(a,"TextOverlayContentHeader",{}),o.TextOverlayWithLogo=n(a,"TextOverlayContentHeaderWithLogo",{isFeatured:!0}),o.TextOverlayCenterFullBleedGradient=n(a,"TextOverlayCenterFullBleedGradient",{contentAlign:"center",background:"gradient"}),o.SplitScreenImgLeft=n(o,"SplitScreenImgLeft",{contentAlign:"left",contentPosition:"end"}),o.SplitScreenImgRight=n(o,"SplitScreenImgRight",{contentAlign:"left",contentPosition:"start"}),o.SplitScreenImgRightContentCenter=n(o,"SplitScreenImgRightContentCenter",{contentAlign:"center",contentPosition:"start",contentStyle:"card",mediaWidth:"grid"}),o.AssetMiddleCenterBig=n(o,"AssetMiddleCenterBig",{assetPosition:"middle",contentAlign:"center",contentPosition:"above",mediaWidth:"small"}),o.AssetMiddleCenterFullBleed=n(o,"AssetMiddleCenterFullBleed",{assetPosition:"middle",contentAlign:"center",contentPosition:"above",mediaWidth:"grid"}),o.SubjectFocus=n(o,"SubjectFocus",{contentAlign:"left",contentPosition:"above",hasExtraSpaceBetweenSeparator:!0,leadRailAnchor:!0,mediaWidth:"small",publishDatePosition:"top",reducedSpacings:!0},{showContributorImage:!1}),o.SplitScreenImageRightFullBleed=n(r,"SplitScreenImageRightFullBleed",{},{isTextRight:!1,showContributorImage:!1}),o.SplitScreenImageRightInset=n(r,"SplitScreenImageRightInset",{},{isInset:!0,isTextRight:!1,showContributorImage:!1}),o.SplitScreenImageLeftFullBleed=n(r,"SplitScreenImageLeftFullBleed",{},{isTextRight:!0,showContributorImage:!1}),o.SplitScreenImageLeftInset=n(r,"SplitScreenImageLeftInset",{},{isInset:!0,isTextRight:!0,showContributorImage:!1}),o.BusinessContentHeader=n(o,"BusinessContentHeader",{contentAlign:"center",contentPosition:"below",mediaWidth:"fullbleed",showContentDivider:!0,hideRubric:!0}),o.PodcastContentHeader=n(o,"PodcastContentHeader",{contentAlign:"center",contentPosition:"below",shouldUseSmallLede:!0,showPodcastButton:!0,hasDivider:!0},{hideLedeCaption:!0}),e.exports=o},1431:function(e,t,i){e.exports=i(1442)},1432:function(e,t,i){const{asConfiguredComponent:n}=i(9),o=i(1454);e.exports=n(o,"ReviewRatingData")},1433:function(e,t,i){e.exports=i(1422)},1441:function(e,t,i){const n=i(8),o=i(1),a=i(0),r=i(136),{withLightbox:l}=i(1431),s=i(75),{trackComponent:d}=i(5),c=i(21),p=i(46),g=i(37),u=i(178),m=i(91),h=i(109),y=i(11),b=i(1447),f=i(179),C=i(1453),{useNativeShare:v}=i(93),{getThemedBylineVariation:w}=i(253),{ContentHeaderSocialIcons:S}=i(447),{ContentHeaderWrapper:k,ContentHeaderOffersData:x,ContentHeaderLeadOverride:$,ContentHeaderAccreditationBottom:T,ContentHeaderContainer:E,ContentHeaderPublishDate:N,ContentHeaderLeadAssetWrapper:B,ContentHeaderBylines:L,ContentHeaderPersistentAside:I,PodcastContentHeaderDivider:A,ContentHeaderHedAccreditationWrapper:P}=i(251),{ContentHeaderDek:R}=i(247),{getOverrideBehaviour:D}=i(437),M=({additionalPhotos:e,authorsTitleBlockPosition:t,awards:i,business:o,bylineVariation:r,captionStyle:s,captionWidth:c,className:p,contentHeaderCategories:g,contentSponsorNames:u,contributorImage:m,contributors:h,ctaText:y,dangerousDek:M,dangerousHed:O,dividerType:H,externalLinks:_,hasDesktopTitleBlockDivider:G,hasLedeLightboxButton:W,hasLightbox:j,hasSlideshow:F,hasStaticPositionedAward:V,hasStickyBoxIndexPosition:U,hasStickySocialIcons:z,hideByLine:q,hideContributors:K,hideDangerousDek:Z,hideRubric:X,hideSocialIconsOnMobile:Y,hideTopDisclaimerOnMobile:J,hideTopRating:Q,hidePublishDate:ee,interactiveOverride:te,isBusinessContentHeader:ie,isLiveStoryType:ne,isStoryLive:oe,issueDate:ae,issueLink:re,lede:le,ledeSocialIcons:se,offers:de,persistentAsideAlign:ce,price:pe,publishDate:ge,rating:ue,reviewRating:me,rubric:he,rubricVariation:ye,itemsCount:be,hasContributorImageBackground:fe,metadataVideo:Ce,showContentDivider:ve,showContributorImage:we,showSponsorBlock:Se,showHeaderButton:ke,hideIssueDate:xe,hideIssueDatePipeSeparator:$e,hideLede:Te,hideLedeCaption:Ee,hasDisabledCloseOnClickForLightbox:Ne,hasNarrowHeader:Be,podcastPagePrimaryCta:Le,shouldDisplayPremiereDate:Ie,shouldShortenHeadline:Ae,showFullWidthLeadImage:Pe,showIssueCopyByDate:Re,showTextOverlayDek:De,slideShowVariation:Me,socialIconsToHide:Oe,socialMedia:He,socialTitle:_e,socialDescription:Ge,stickySocialAnchorBottom:We,stickySocialAnchorTop:je,theme:Fe,sponsoredContentHeaderProps:Ve,sponsorByline:Ue,variations:{assetPosition:ze,copyWidth:qe,contentAlign:Ke,contentPosition:Ze,contentStyle:Xe,hasExtraSpaceBetweenSeparator:Ye=!1,hasLedeLightbox:Je,hasNoRowPadding:Qe,hasInlinePublishDate:et,hasXsNavSpacing:tt,ledeAlign:it,leadRailAnchor:nt,mediaWidth:ot,publishDatePosition:at="bottom",reducedSpacings:rt=!1,hasDivider:lt,showPodcastButton:st,shouldUseSmallLede:dt},hasNativeShareButton:ct,shouldEnableNativeShareOnDesktop:pt,showBreadCrumb:gt,venueAwards:ut,hasInvertedCaption:mt,hasInvertedLedeBackground:ht,variationName:yt})=>{a.useEffect(()=>{d("ContentHeader",yt)},[yt]);const bt="middle"===ze,ft=(e=>e&&1===Object.keys(e).length&&e.author&&1===e.author.items.length)(h)&&et,Ct="storyimage"===D(te),vt=w({bylineVariation:r,theme:Fe}),wt=a.createElement(b,{authorsPosition:t,business:o,bylineVariation:vt,contentHeaderCategories:g,contentSponsorNames:u,contributors:h,dangerousHed:O,dividerType:H,externalLinks:_,hasContentDivider:ve,hasDesktopTitleBlockDivider:G,hasDivider:lt,hasNoRowPadding:Qe,hasExtraSpaceBetweenSeparator:Ye,hideIssueDate:xe,hideIssueDatePipeSeparator:$e,hideRubric:X,hidePublishDate:ee,isBusinessContentHeader:ie,isLiveStoryType:ne,isStoryLive:oe,issueDate:ae,issueLink:re,itemsCount:be,metadataVideo:Ce,podcastPagePrimaryCta:Le,price:pe,publishDate:ge,publishDatePosition:at,rubric:he,rubricVariation:ye,showIssueCopyByDate:Re,showPodcastButton:st,theme:Fe});let St=f;j&&Je&&(St=e?l(f,e,F,Me,Ne):l(f,[le]));const{showNativeShareButton:kt,pageUrl:xt}=v(ct,pt),$t="hidden"!==t,Tt=h&&Object.keys(h).length>1,Et=h&&!K&&a.createElement(L,{contributors:h,bylineVariation:vt,isCompact:!1,inlinePublishDate:ft}),Nt=!ee&&a.createElement(N,{inlinePublishDate:ft,"data-testid":"ContentHeaderPublishDate",mediaWidth:ot,contentAlign:Ke},ge),Bt=a.createElement(a.Fragment,null,a.createElement(C,{bylinesBlock:Et,contributorImage:m,dangerousDek:M,hideDangerousDek:Z,dividerType:H,hasContributorImageBackground:fe,hasDesktopTitleBlockDivider:G,hasLede:!!le,hideTopRating:Q,hasStickySocialIcons:z,hideByLine:q,hideSocialIconsOnMobile:Y,isBusinessContentHeader:ie,isMiddleImage:bt,isSponsoredContent:u.length>0,isLiveStoryType:ne,isStoryLive:oe,metadataVideo:Ce,publishDateBlock:Nt,publishDatePosition:at,rating:ue,reviewRating:me,shouldDisplayPremiereDate:Ie,shouldShowAuthorsInTitleBlock:$t,showContributorImage:we,showSponsorBlock:Se,socialIconsToHide:Oe,socialMedia:He,socialTitle:_e,socialDescription:Ge,sponsorByline:Ue,sponsoredContentHeaderProps:Ve,pageUrl:xt,showNativeShareButton:kt,venueAwards:ut,mediaWidth:ot,contentAlign:Ke,contentPosition:Ze,theme:Fe}),a.createElement(x,{ctaText:y,hideTopDisclaimerOnMobile:J,offers:de,showHeaderButton:ke,buttonPosition:"content-header"}));let Lt;return Lt=ie?a.createElement(P,null,O&&wt,Bt):a.createElement(a.Fragment,null,O&&wt,Bt),a.createElement(k,{className:n("content-header",{[p]:p}),isLiveStoryType:ne,publishDatePosition:at,hasXsNavSpacing:tt,contentAlign:Ke,assetPosition:ze,shouldShowAuthorsInTitleBlock:$t,captionStyle:s,copyWidth:qe,mediaWidth:ot,contentStyle:Xe,contentPosition:Ze,isBusinessContentHeader:ie,shouldShortenHeadline:Ae,reducedSpacings:rt,hasInvertedCaption:mt,containerTheme:Fe,shouldBylineContentStacked:Tt,hasExtraSpaceBetweenSeparator:Ye,hasLede:!!le,hasNarrowHeader:Be,showBreadCrumb:gt,showTextOverlayDek:De},a.createElement(E,{containerTheme:Fe,mediaWidth:ot,showFullWidthLeadImage:Pe,contentStyle:Xe,contentPosition:Ze,"data-testid":"ContentHeaderContainer"},Lt,Ct?a.createElement($,{contentPosition:Ze,dangerouslySetInnerHTML:{__html:te.markup}}):!Te&&le&&a.createElement(B,{awards:i,hasLightboxButton:W,hasStaticPositionedAward:V,hideLedeCaption:Ee,hasDisabledCloseOnClickForLightbox:Ne,lede:le,captionWidth:c,shouldRenderRailAnchor:nt,shouldUseSmallLede:dt,socialIcons:se,mediaWidth:ot,containerTheme:Fe,hasInvertedLedeBackground:ht,ledeAlign:it,showFullWidthLeadImage:Pe,isBusinessContentHeader:ie,as:St,className:n({["lead-asset--width-"+ot]:ot})}),!Z&&M&&bt&&a.createElement(T,null,a.createElement(R,{dangerouslySetInnerHTML:{__html:M},assetPosition:ze,mediaWidth:ot,"data-testid":"ContentHeaderDek"})),lt&&a.createElement(A,null)),!kt&&z&&He&&a.createElement(I,{attributes:{shouldFadeOnMove:!0},align:ce,anchorBottom:We,anchorTop:je,hasStickyBoxIndexPosition:U,fullWidthSelector:".container--full, .full-bleed-ad, .callout--feature-large"},a.createElement(S,Object.assign({},He,{className:"social-icons--share"}))))};M.propTypes={additionalPhotos:o.array,authorsTitleBlockPosition:o.oneOf(["above","below","hidden"]),awards:o.array,business:o.shape({address:o.object,phone:o.string,email:o.string,socialMedia:o.array}),bylineVariation:o.string,captionStyle:o.oneOf(["default","span-content-well"]),captionWidth:o.oneOf(["standard","fullbleed"]),className:o.string,contentHeaderCategories:o.shape({title:o.string,tags:o.array}),contentSponsorNames:o.array,contributorImage:o.shape(y.propTypes),contributors:o.shape(g.propTypes.contributors),ctaText:o.string,dangerousDek:o.string,dangerousHed:o.string,dividerType:o.oneOf(["both","bottom","top"]),externalLinks:o.array,hasContributorImageBackground:o.bool,hasDesktopTitleBlockDivider:o.bool,hasDisabledCloseOnClickForLightbox:o.bool,hasInvertedCaption:o.bool,hasInvertedLedeBackground:o.bool,hasLedeLightboxButton:o.bool,hasLightbox:o.bool,hasNarrowHeader:o.bool,hasNativeShareButton:o.bool,hasSlideshow:o.bool,hasStaticPositionedAward:o.bool,hasStickyBoxIndexPosition:o.bool,hasStickySocialIcons:o.bool,hideByLine:o.bool,hideContributors:o.bool,hideDangerousDek:o.bool,hideIssueDate:o.bool,hideIssueDatePipeSeparator:o.bool,hideLede:o.bool,hideLedeCaption:o.bool,hidePublishDate:o.bool,hideRubric:o.bool,hideSocialIconsOnMobile:o.bool,hideTopDisclaimerOnMobile:o.bool,hideTopRating:o.bool,interactiveOverride:o.shape({markup:o.string,behavior:o.string}),isBusinessContentHeader:o.bool,isLiveStoryType:o.bool,isStoryLive:o.bool,issueDate:o.string,issueLink:o.string,itemsCount:o.shape(s.propTypes),lede:o.oneOfType([o.shape(y.propTypes),o.shape(u.propTypes),o.shape(m.propTypes)]),ledeSocialIcons:o.shape(c.propTypes),metadataVideo:o.shape({isLive:o.bool,premiereDate:o.string,premiereGap:o.number,series:o.string,videoLength:o.number}),offers:o.array,persistentAsideAlign:o.oneOf(["left","left-lead-asset"]),podcastPagePrimaryCta:o.string,price:o.string,publishDate:o.string.isRequired,rating:o.shape(h.propTypes),reviewRating:o.number,rubric:o.shape(p.propTypes),rubricVariation:o.string,shouldDisplayPremiereDate:o.bool,shouldEnableNativeShareOnDesktop:o.bool,shouldShortenHeadline:o.bool,showBreadCrumb:o.bool,showContentDivider:o.bool,showContributorImage:o.bool,showFullWidthLeadImage:o.bool,showHeaderButton:o.bool,showIssueCopyByDate:o.bool,showSponsorBlock:o.bool,showTextOverlayDek:o.bool,slideShowVariation:o.string,socialDescription:o.string,socialIconsToHide:o.array,socialMedia:o.shape(c.propTypes),socialTitle:o.string,sponsorByline:o.string,sponsoredContentHeaderProps:o.shape({campaignUrl:o.string,sponsorLogo:o.shape(y.propTypes),sponsorName:o.string}),stickySocialAnchorBottom:r.propTypes.anchorBottom,stickySocialAnchorTop:r.propTypes.anchorTop,theme:o.oneOf(["standard","inverted","special","live"]),variationName:o.string,variations:o.shape({assetPosition:o.oneOf(["bottom","middle"]),contentAlign:o.oneOf(["center","left"]),contentPosition:o.oneOf(["above","below","start","end"]),contentStyle:o.oneOf(["card","dense-card","item"]),copyWidth:o.oneOf(["grid","fullbleed"]),hasDivider:o.bool,hasExtraSpaceBetweenSeparator:o.bool,hasInlinePublishDate:o.bool,hasLedeLightbox:o.bool,hasNoRowPadding:o.bool,hasXsNavSpacing:o.bool,leadRailAnchor:o.bool,ledeAlign:o.oneOf(["default","center"]),mediaWidth:o.oneOf(["small","smallrule","grid","fullbleed"]),publishDatePosition:o.oneOf(["top","bottom"]),reducedSpacings:o.bool,shouldUseSmallLede:o.bool,showPodcastButton:o.bool}),venueAwards:o.string},M.defaultProps={authorsTitleBlockPosition:"hidden",business:{address:{},phone:"",email:"",socialMedia:[],link:""},captionStyle:"default",captionWidth:"standard",contentSponsorNames:[],dividerType:"both",externalLinks:[],hasContributorImageBackground:!1,hasDesktopTitleBlockDivider:!1,hasInvertedCaption:!1,hasInvertedLedeBackground:!0,hasLightbox:!1,hasNarrowHeader:!1,hasNativeShareButton:!1,hasSlideshow:!1,hasStaticPositionedAward:!1,hasStickySocialIcons:!0,hideByLine:!1,hideContributors:!1,hideDangerousDek:!1,hideIssueDatePipeSeparator:!1,hideLede:!1,hideLedeCaption:!1,hidePublishDate:!1,hideRubric:!1,hideSocialIconsOnMobile:!1,isLiveStoryType:!1,isStoryLive:!1,metadataVideo:{},persistentAsideAlign:"left",shouldDisplayPremiereDate:!1,shouldEnableNativeShareOnDesktop:!1,shouldShortenHeadline:!1,showContentDivider:!1,showContributorImage:!0,showFullWidthLeadImage:!1,showIssueCopyByDate:!1,showSponsorBlock:!1,showTextOverlayDek:!1,socialIconsToHide:[],stickySocialAnchorBottom:{selector:".page",edge:"bottom"},stickySocialAnchorTop:{selector:"[data-testid='ContentHeaderContainer']",edge:"bottom"},theme:"standard",variations:{contentAlign:"center",contentPosition:"above",hasDivider:!1,hasExtraSpaceBetweenSeparator:!1,hasInlinePublishDate:!1,hasLedeLightbox:!1,hasNoRowPadding:!1,hasXsNavSpacing:!1,leadRailAnchor:!1,ledeAlign:"default",publishDatePosition:"bottom",reducedSpacings:!1,shouldUseSmallLede:!1,showPodcastButton:!1}},M.displayName="ContentHeader",e.exports=M},1442:function(e,t,i){const n=i(0),o=i(154),a=i(8),{createGlobalStyle:r}=i(3),l=i(1443),s=i(45),d=i(1578),c=i(1412),p=i(1411),{useIntl:g}=i(2),u=i(1445).default,m=i(440),h=i(1446),y=i(32),{trackComponent:b}=i(5),{LightboxCloseButtonIcon:f,LightboxSwipe:C,LightboxWrapper:v,LightboxSlidesWrapper:w}=i(498),{getColorToken:S,getZIndex:k}=i(4);let x;const $=r`
  .lightbox {
    height: 100vh;
  }

  .body__lightbox--open {
    overflow: hidden;
  }

  .lightbox__overlay {
    position: fixed;
    top: 0;
    left: 0;
    transition: opacity 0.2s;
    opacity: 0;
    z-index: ${k("hyperstitialLayer")};
    background-color: ${({theme:e})=>S(e,"colors.consumption.lead.standard.background")};

    &.lightbox__overlay--open {
      opacity: 1;

      &.lightbox__overlay--closing {
        opacity: 0;
      }
    }
  }

  .listicle-variation {
    overflow-y: scroll;
  }
`;e.exports={withLightbox:(e,t,i,r,S=!1,k)=>T=>{const[E,N]=n.useState(-1),{formatMessage:B}=g(),L=e=>i&&N(l(E+e,0,t.length-1)),I=E>-1,A=e=>{var t;"swipeClose"!==e&&("IMG"===(t=e.target).tagName||"SPAN"===t.tagName)||N(void 0)},P={transform:`translate(${"listicle"===r?"0":-100*E/t.length}%)`},R=()=>{x&&(x.style.height=window.innerHeight+"px")};n.useEffect(()=>{b("LightBox",k)}),n.useEffect(()=>{const e=s(R,50);return window.addEventListener("resize",e),()=>{window.removeEventListener("resize",e)}}),n.useEffect(()=>{!I&&x&&p.enableBodyScroll(x)},[I]);const D=(e,t)=>S?{}:t.reduce((t,i)=>Object.assign(Object.assign({},t),{[i]:t=>A(e||t)}),{});return n.createElement(v,null,n.createElement(m.Provider,{value:{expandHandler:e=>N(d(t,{id:e})),isInSlides:e=>c(t,{id:e})}},n.createElement(e,Object.assign({},T))),n.createElement(o,{className:a("lightbox "+(r?r+"-variation":"")),overlayClassName:{base:a("lightbox__overlay"),afterOpen:"lightbox__overlay--open",beforeClose:"lightbox__overlay--closing"},bodyOpenClassName:"body__lightbox--open",isOpen:I,onAfterOpen:()=>{p.disableBodyScroll(x),R()},onRequestClose:()=>{N(void 0)},contentRef:e=>{e&&(x=e)},closeTimeoutMS:200,contentLabel:B(u.contentLabel)},n.createElement(f,{ButtonIcon:y,isIconButton:!0,className:r?r+"-variation-close":"",dataAttrs:{"data-action":"close"},label:B(u.closeButtonIconLabel),onClickHandler:A,role:"button",ariaLabel:B(u.closeButtonIconLabel)}),n.createElement(C,Object.assign({},D("swipeClose",["onSwipeDown","onSwipeUp"]),{onSwipeRight:()=>L(-1),onSwipeLeft:()=>L(1),shouldPreventDefaultEvent:!0}),n.createElement(w,Object.assign({className:r?r+"-variation-slide-wrapper":"",style:P},D("",["onMouseDown","onTouchEnd"])),t.map(e=>n.createElement(h,Object.assign({},e,{className:r?r+"-variation-slide":"",key:e.id}))))),n.createElement($,null)))}}},1443:function(e,t,i){var n=i(1444),o=i(1569);e.exports=function(e,t,i){return void 0===i&&(i=t,t=void 0),void 0!==i&&(i=(i=o(i))==i?i:0),void 0!==t&&(t=(t=o(t))==t?t:0),n(o(e),t,i)}},1444:function(e,t){e.exports=function(e,t,i){return e==e&&(void 0!==i&&(e=e<=i?e:i),void 0!==t&&(e=e>=t?e:t)),e}},1445:function(e,t,i){Object.defineProperty(t,"__esModule",{value:!0});const n=i(2);t.default=n.defineMessages({contentLabel:{id:"Lightbox.ContentLabel",defaultMessage:"Photo Gallery",description:"Lightbox component content label"},closeButtonIconLabel:{id:"Lightbox.CloseButtonIconLabel",defaultMessage:"Close Lightbox",description:"Lightbox component close button icon label"}})},1446:function(e,t,i){var n=this&&this.__rest||function(e,t){var i={};for(var n in e)Object.prototype.hasOwnProperty.call(e,n)&&t.indexOf(n)<0&&(i[n]=e[n]);if(null!=e&&"function"==typeof Object.getOwnPropertySymbols){var o=0;for(n=Object.getOwnPropertySymbols(e);o<n.length;o++)t.indexOf(n[o])<0&&Object.prototype.propertyIsEnumerable.call(e,n[o])&&(i[n[o]]=e[n[o]])}return i};const o=i(1),a=i(0),r=i(67),l=i(11),s=i(49),{LightboxSlideWrapper:d,LightboxSlideTopSpacer:c,LightboxSlideImageCaptionOuter:p,LightboxSlideCaptionContainer:g}=i(498),u=e=>{var{caption:t,className:i,credit:o,dangerousCredit:s,dangerousCaption:u}=e,m=n(e,["caption","className","credit","dangerousCredit","dangerousCaption"]);const[h,y]=a.useState("landscape");return a.createElement(d,{className:i,screenOrientation:h},a.createElement(c,null),a.createElement(p,null,a.createElement(l,Object.assign({},m,{contentType:"photo",onAssetLoaded:({width:e,height:t})=>{e<t?y("portrait"):e>t?y("landscape"):e===t&&y("square")}})),a.createElement(g,null,a.createElement(r,{dangerousCaptionText:t||u,dangerousCredit:o||s,hasLinebreak:"portrait"===h}))))};u.propTypes=Object.assign(Object.assign({},s.propTypes),{caption:o.string,credit:o.string,dangerousCaption:o.string,dangerousCredit:o.string}),e.exports=u},1447:function(e,t,i){const n=i(1),o=i(0),{asConfiguredComponent:a}=i(9),r=i(75),l=i(46),s=i(499),d=i(500),c=i(1448),p=i(501),g=i(1452),u=i(128),{ContentHeaderContentDivider:m,ContentHeaderLowerBylineDateBlock:h,ContentHeaderItemCount:y,ContentHeaderHed:b,ContentHeaderTitleBlockWrapper:f,PodcastButtonWrapper:C,PodcastButton:v,PodcastLink:w}=i(176),S=e=>e.length>0?e[0]:null,k=({authorsPosition:e,business:t,bylineVariation:i,contentHeaderCategories:n,contentSponsorNames:a,contributors:r,dangerousHed:l,dividerType:k,externalLinks:x,hasContentDivider:$,hasDesktopTitleBlockDivider:T,hasExtraSpaceBetweenSeparator:E,hideIssueDate:N,hideIssueDatePipeSeparator:B,hasNoRowPadding:L,hidePublishDate:I,hideRubric:A,isBusinessContentHeader:P,isLiveStoryType:R,isStoryLive:D,issueDate:M,issueLink:O,itemsCount:H,metadataVideo:_,price:G,podcastPagePrimaryCta:W,publishDate:j,publishDatePosition:F,rubric:V,rubricVariation:U,shouldDisplayLiveIndicator:z,showIssueCopyByDate:q,showItemCount:K,showPodcastButton:Z,theme:X})=>{const Y=$&&"above"!==e,J="inverted"===X,Q=J?"outlined":"filled",ee=T&&("both"===k||"top"===k);return o.createElement(f,{rowWithTopBorder:ee,isBusinessContentHeader:P,"data-testid":"ContentHeaderTitleBlockWrapper"},_.isLive&&z&&o.createElement(u,null),o.createElement(c,{authorsPosition:e,bylineVariation:i,contributors:r,hasExtraSpaceBetweenSeparator:E,hasNoRowPadding:L,hideIssueDate:N,hideIssueDatePipeSeparator:B,hidePublishDate:I,hideRubric:A,isLiveStoryType:R,isStoryLive:D,issueDate:M,issueLink:O,price:G,publishDate:j,publishDatePosition:F,rubric:V,rubricVariation:U,showIssueCopyByDate:q,sponsorName:S(a),theme:X}),o.createElement(p,Object.assign({},n)),o.createElement(b,{dangerouslySetInnerHTML:{__html:l},"data-testid":"ContentHeaderHed"}),"below"===e&&o.createElement(h,null,o.createElement(s,{bylineVariation:i,contributors:r}),o.createElement(d,{hasExtraSpaceBetweenSeparator:E,hidePublishDate:I,publishDate:j})),K&&H&&o.createElement(y,Object.assign({},H)),Y&&o.createElement(m,null),P&&o.createElement(g,{address:null==t?void 0:t.address,phone:null==t?void 0:t.phone,email:null==t?void 0:t.email,socialMedia:null==t?void 0:t.socialMedia,link:null==t?void 0:t.url}),Z&&W&&o.createElement(C,null,o.createElement(v,{btnStyle:Q,href:W,label:"Start Listening Now",target:"blank",inputKind:"link"}),x.length>0&&o.createElement(w,{isInverted:J,href:x[0].url,target:"blank"},"Or, choose where to Listen")))};k.propTypes={authorsPosition:n.oneOf(["above","below","hidden"]),business:n.shape({address:n.object,phone:n.string,email:n.string,socialMedia:n.array,url:n.string}),bylineVariation:n.string,contentHeaderCategories:n.shape({title:n.string,tags:n.array,hasCategoryEyebrow:n.boolean}),contentSponsorNames:n.array,contributors:n.object,dangerousHed:n.string.isRequired,dividerType:n.string,externalLinks:n.array,hasContentDivider:n.bool,hasDesktopTitleBlockDivider:n.bool,hasDivider:n.bool,hasExtraSpaceBetweenSeparator:n.bool,hasNoRowPadding:n.bool,hideIssueDate:n.bool,hideIssueDatePipeSeparator:n.bool,hidePublishDate:n.bool,hideRubric:n.bool,isBusinessContentHeader:n.bool,isLiveStoryType:n.bool,isStoryLive:n.bool,issueDate:n.string,issueLink:n.string,itemsCount:n.shape(r.propTypes),metadataVideo:n.shape({isLive:n.bool,premiereDate:n.string,series:n.string,videoLength:n.number}),podcastPagePrimaryCta:n.string,price:n.string,publishDate:n.string,publishDatePosition:n.oneOf(["top","bottom"]),rubric:n.shape(l.propTypes),rubricVariation:n.string,shouldDisplayLiveIndicator:n.bool,showIssueCopyByDate:n.bool,showItemCount:n.bool,showPodcastButton:n.bool,theme:n.oneOf(["standard","inverted","special"])},k.defaultProps={authorsPosition:"hidden",contentHeaderCategories:{hasCategoryEyebrow:!1},contentSponsorNames:[],dividerType:"both",hasDesktopTitleBlockDivider:!1,hasExtraSpaceBetweenSeparator:!1,hasNoRowPadding:!1,hideIssueDate:!0,hideIssueDatePipeSeparator:!1,hidePublishDate:!1,hideRubric:!1,isBusinessContentHeader:!1,metadataVideo:{},publishDatePosition:"bottom",shouldDisplayLiveIndicator:!1,showIssueCopyByDate:!1,showItemCount:!1,showPodcastButton:!1,theme:"standard"},k.displayName="TitleBlock",e.exports=a(k,"TitleBlock"),e.exports.TitleBlock=k},1448:function(e,t,i){const n=i(1),o=i(0),a=i(128),r=i(46),l=i(499),s=i(1449),d=i(500),{ContentHeaderSponsorName:c,ContentHeaderRubricBlock:p,ContentHeaderRubricDateBlock:g,ContentHeaderRubricPrice:u,ContentHeaderRubricContainer:m,ContentHeaderLiveIndicator:h}=i(166),y=({authorsPosition:e,bylineVariation:t,contributors:i,hasExtraSpaceBetweenSeparator:n,hideIssueDate:y,hideIssueDatePipeSeparator:b,hidePublishDate:f,hideRubric:C,isLiveStoryType:v,isStoryLive:w,issueDate:S,issueLink:k,price:x,publishDate:$,publishDatePosition:T,rubric:E,rubricVariation:N,showIssueCopyByDate:B,sponsorName:L})=>{const I=r[N]||r,A=!y&&S&&E,P=("above"===e||"top"===T)&&!L;return v&&w?o.createElement(h,null,o.createElement(a,{hasBackground:!0,isDiscovery:!1,shouldEnableAnimation:!0})):!(E||S||i)||C?null:(E||S||i)&&o.createElement(p,{hasIssueDateAndRubricBlock:A,"data-testid":"ContentHeaderRubric"},"above"===e&&o.createElement(l,{bylineVariation:t,contributors:i}),o.createElement(g,{"data-testid":"ContentHeaderRubricDateBlock"},E&&o.createElement(m,Object.assign({},E,{isVerticalAlign:"above"===e||"top"===T||A,as:I})),x&&o.createElement(u,null,"/ ",x),L&&o.createElement(c,{hasExtraSpaceBetweenSeparator:n,items:[{name:L}]}),P&&o.createElement(d,{hasExtraSpaceBetweenSeparator:n,hidePublishDate:f,publishDate:$})),o.createElement(s,{hideIssueDate:y,hideIssueDatePipeSeparator:b,issueDate:S,issueLink:k,showIssueCopyByDate:B}))};y.propTypes={authorsPosition:n.string,bylineVariation:n.string,contributors:n.object,hasExtraSpaceBetweenSeparator:n.bool,hasNoRowPadding:n.bool,hideIssueDate:n.bool,hideIssueDatePipeSeparator:n.bool,hidePublishDate:n.bool,hideRubric:n.bool,isLiveStoryType:n.bool,isStoryLive:n.bool,issueDate:n.string,issueLink:n.string,price:n.string,publishDate:n.string,publishDatePosition:n.string,rubric:n.shape(r.propTypes),rubricVariation:n.string,showIssueCopyByDate:n.bool,sponsorName:n.string},y.defaultProps={isLiveStoryType:!1,isStoryLive:!1},e.exports=y},1449:function(e,t,i){const n=i(1),o=i(0),{ContentHeaderRubricIssueDate:a}=i(166),r=({hideIssueDate:e,hideIssueDatePipeSeparator:t,issueDate:i,issueLink:n,showIssueCopyByDate:r})=>e||!i?null:o.createElement(a,{name:`${i}${r?" Issue":""}`,url:n,hideIssueDatePipeSeparator:t});r.propTypes={hideIssueDate:n.bool,hideIssueDatePipeSeparator:n.bool,issueDate:n.string,issueLink:n.string,showIssueCopyByDate:n.bool},e.exports=r},1450:function(e,t,i){const n=i(1),o=i(0),{CategoriesWrapper:a,CategoriesTitle:r,CategoriesItemList:l,CategoriesItem:s,CategoriesLink:d}=i(1451),c=({title:e,tags:t})=>o.createElement(a,null,o.createElement(r,null,e),o.createElement(l,null,t.map(e=>o.createElement(s,{key:e.name},o.createElement(d,{href:e.slug},e.name)))));c.propTypes={tags:n.array,title:n.string},c.defaultProps={tags:[]},c.displayName="Categories",e.exports=c},1451:function(e,t,i){const{default:n}=i(3),{BaseText:o,BaseLink:a}=i(10),{calculateSpacing:r,getColorToken:l,getLinkStyles:s}=i(4),d=n.div.withConfig({displayName:"CategoriesWrapper"})`
  display: flex;
  flex-wrap: wrap;
  align-items: baseline;
  margin-top: ${r(.5)};
`,c=n(o).withConfig({displayName:"CategoriesTitle"})`
  margin-right: ${r(1)};
`;c.defaultProps={as:"div",colorToken:"colors.consumption.lead.standard.accreditation",typeIdentity:"typography.definitions.globalEditorial.accreditation-feature"};const p=n.ul.withConfig({displayName:"CategoriesItemList"})`
  margin: 0;
  padding: 0;
  line-height: 1;
`,g=n.li.withConfig({displayName:"CategoriesItem"})`
  display: inline-block;

  &:not(:last-child) {
    &::after {
      margin: 0 ${r(1)};
      color: ${({theme:e})=>l(e,"colors.consumption.lead.standard.divider")};
      content: '|';
    }
  }
`,u=n(a).withConfig({displayName:"CategoriesLink"})`
  ${({theme:e})=>s(e,"colors.consumption.lead.standard.link",null,"global")};
`;u.defaultProps={typeToken:"typography.definitions.globalEditorial.accreditation-core"},e.exports={CategoriesWrapper:d,CategoriesTitle:c,CategoriesItemList:p,CategoriesItem:g,CategoriesLink:u}},1452:function(e,t,i){const n=i(0),o=i(1),{AddressBlockWrapper:a,HeaderContactInformation:r,HeaderSocialInformation:l}=i(176),s=i(242),d=i(243),c=({address:e,email:t,phone:i,socialMedia:o,link:c})=>{if(!e&&!i&&!t)return null;const{street:p,city:g,state:u,postalCode:m,country:h}=e,y=[];[p,g,u,m,h].forEach(e=>{e&&y.push(e)});let b=c.replace(/^(https?:|)\/\//,"");return b=b.includes("www.")?b:"www."+b,n.createElement(a,null,e&&n.createElement("div",{"data-testid":"HeaderAddressDetails"},n.createElement("div",{className:"StreetAndCity"},p&&n.createElement("span",{"data-testid":"HeaderAddressStreet"},p),g&&n.createElement("span",{"data-testid":"HeaderAddressCity"},y[0]!==g&&", ",g,y[0]!==u&&", ")),n.createElement("div",{className:"StatePostalAndCountry"},u&&n.createElement("span",{"data-testid":"HeaderAddressState"},u),m&&n.createElement("span",{"data-testid":"HeaderAddressPostalCode"},y[0]!==m&&", ",m),h&&n.createElement("span",{"data-testid":"HeaderAddressCountry"},y[0]!==h&&", ",h))),n.createElement(r,null,i&&n.createElement("div",{"data-testid":"HeaderAddressPhone",dangerouslySetInnerHTML:{__html:i}}),c&&n.createElement("a",{"data-testid":"HeaderAddressWebUrl",href:c,"aria-label":c,rel:"nofollow noopener noreferrer",target:"_blank"},b)),n.createElement(l,null,t&&n.createElement("a",{"data-testid":"HeaderAddressEmail",href:"mailto:"+t,"aria-label":t,rel:"nofollow noopener noreferrer",target:"_blank"},s()),(null==o?void 0:o.length)>0&&n.createElement("a",{"data-testid":"HeaderAddressInstagram",href:o[0].handle,"aria-label":o[0].network,rel:"nofollow noopener noreferrer",target:"_blank"},d())))};c.propTypes={address:o.object,email:o.string,link:o.string,phone:o.string,socialMedia:o.array},e.exports=c},1453:function(e,t,i){const n=i(0),{useIntl:o}=i(2),a=i(1),r=i(502),l=i(1432),s=i(11),d=i(109),c=i(21),p=i(165),g=i(164).default,u=i(503),m=i(441),{ContentHeaderByline:h,ContentHeaderAccreditationSocialIcons:y,ContentHeaderContributorImage:b,ContentHeaderNativeShareButton:f,ContentHeaderBylineContent:C,ContentHeaderDekRewards:v,ContentHeaderAccreditation:w,ContentHeaderDek:S,SummaryPremiereWrapper:k}=i(247),x=({venueAwards:e})=>n.createElement(n.Fragment,null,e&&n.createElement(v,{dangerouslySetInnerHTML:{__html:e}}));x.propTypes={venueAwards:a.string};const $=({bylinesBlock:e,contributorImage:t,dangerousDek:i,dividerType:a,hasContributorImageBackground:s,hasDesktopTitleBlockDivider:c,hasLede:v,hasStickySocialIcons:$,hideByLine:T,hideDangerousDek:E,hideSocialIconsOnMobile:N,hideTopRating:B,isBusinessContentHeader:L,isMiddleImage:I,isSponsoredContent:A,showSponsorBlock:P,sponsorByline:R,sponsoredContentHeaderProps:D,isLiveStoryType:M,metadataVideo:O,publishDateBlock:H,publishDatePosition:_,rating:G,reviewRating:W,shouldShowAuthorsInTitleBlock:j,showContributorImage:F,socialIconsToHide:V,socialMedia:U,socialTitle:z,socialDescription:q,pageUrl:K,shouldDisplayPremiereDate:Z,showNativeShareButton:X,venueAwards:Y,theme:J})=>{const{rating:Q,count:ee}=G||{},te=x({venueAwards:Y}),ie={showDek:!E&&i&&!I,showVenueAwards:Y,showByline:!j&&!A&&!T,showNativeShareButton:X,socialMedia:U},{isLive:ne,premiereGap:oe,premiereDate:ae}=O,{formatMessage:re}=o();return n.createElement(w,{className:"content-header__accreditation",shouldShowAuthorsInTitleBlock:j,isBusinessContentHeader:L,hasLede:v,rowWithBottomBorder:c&&("both"===a||"bottom"===a),visibilityInfo:ie,"data-testid":"ContentHeaderAccreditation"},ie.showDek&&n.createElement(S,{dangerouslySetInnerHTML:{__html:i},as:"div"}),Z&&ae&&!ne&&n.createElement(k,null,n.createElement(m,{premiereDate:ae,premiereGap:oe,containerTheme:J,hideTimeStampIcon:!0})),M&&P&&n.createElement(u,{sponsorByline:R,sponsoredContentHeaderProps:Object.assign({},D),theme:J}),te,W&&!B&&n.createElement(l,{rating:W}),ie.showByline&&n.createElement(h,{isLiveStoryType:M,sponsorName:null==D?void 0:D.sponsorName,showSponsorBlock:P},F&&t&&n.createElement(b,Object.assign({},t,{sizes:L?"100%":"66px",hasContributorImageBackground:s,isBusinessContentHeader:L})),n.createElement(C,null,e,"bottom"===_&&H)),X?n.createElement(f,null,n.createElement(p,{shareData:{url:K,title:z,text:q}})):U&&n.createElement(y,Object.assign({},U,{className:"content-header__social-share",hideSocialIconsOnMobile:N,hasStickySocialIcons:$,socialIconsToHide:V})),!!Q&&!!ee&&n.createElement(d,{averageRatingCount:Math.round(10*Q)/10,hasBorderTop:!0,link:{label:re(g.readReviews),onClick:e=>{e.preventDefault();const t=document.getElementById("reviews"),{top:i}=r(t);window.scrollTo(0,i-56)},url:"#reviews"},totalRatingCount:ee}))};$.propTypes={bylinesBlock:a.node,contributorImage:a.shape(s.propTypes),dangerousDek:a.string,dividerType:a.string,hasContributorImageBackground:a.bool,hasDesktopTitleBlockDivider:a.bool,hasLede:a.bool,hasStickySocialIcons:a.bool,hideByLine:a.bool,hideDangerousDek:a.bool,hideSocialIconsOnMobile:a.bool,hideTopRating:a.bool,isBusinessContentHeader:a.bool,isLiveStoryType:a.bool,isMiddleImage:a.bool,isSponsoredContent:a.bool,metadataVideo:a.shape({isLive:a.bool,premiereDate:a.string,premiereGap:a.number,series:a.string,videoLength:a.number}),pageUrl:a.string,publishDateBlock:a.node,publishDatePosition:a.string,rating:a.shape(d.propTypes),reviewRating:a.number,shouldDisplayPremiereDate:a.bool,shouldShowAuthorsInTitleBlock:a.bool,showContributorImage:a.bool,showNativeShareButton:a.bool,showSponsorBlock:a.bool,socialDescription:a.string,socialIconsToHide:a.array,socialMedia:a.shape(c.propTypes),socialTitle:a.string,sponsorByline:a.string,sponsoredContentHeaderProps:a.shape({campaignUrl:a.string,sponsorLogo:a.shape(s.propTypes),sponsorName:a.string}),theme:a.oneOf(["standard","inverted","special"]),venueAwards:a.string},$.defaultProps={hideByLine:!1,hideDangerousDek:!1,isLiveStoryType:!1,metadataVideo:{},shouldDisplayPremiereDate:!1,showSponsorBlock:!1,socialIconsToHide:[],theme:"standard"},e.exports=$},1454:function(e,t,i){const n=i(1),o=i(8),a=i(0),{useIntl:r}=i(2),l=i(439),s=i(14),d=i(99),{trackComponent:c}=i(5),p=i(1455).default,{ReviewRatingDataLabel:g,ReviewRatingDataWrapper:u,ReviewRatingDataValue:m,ReviewRatingDataExplainer:h,ReviewRatingDataExplainerModal:y,ReviewRatingDataExplainerModalList:b,ReviewRatingDataExplainerModalListRating:f,ReviewRatingDataExplainerModalListDescribe:C}=i(1456),v=({className:e,rating:t,ratingList:i})=>{a.useEffect(()=>{c("ReviewRatingData")},[]);const n=r(),[v,w]=a.useState(!0),S=()=>{w(!v)},k=a.createElement(y,{className:"review-rating-data__rating-explainer-modal"},i.map((e,t)=>a.createElement(b,{key:e},a.createElement(f,{as:"span"},t+1),a.createElement(C,{as:"span"},e))));return a.createElement(u,{className:o("review-rating-data",e)},a.createElement(g,{className:"review-rating-data__label"},n.formatMessage(p.dataLabel)),a.createElement(m,null,t,"/10"),a.createElement(h,null,a.createElement(s.Utility,{isIconButton:!0,ButtonIcon:l,className:"review-rating-data__info-button",label:n.formatMessage(p.buttonLabel),onClickHandler:S,role:"button"}),!v&&a.createElement(d,{className:"alert__tooltip",arrowPosition:52,gaIdentifier:"review-rating-explainer",isVisible:!0,isTooltip:!0,onClose:S,shouldUseArrow:!0},k)))};v.propTypes={className:n.string,rating:n.number,ratingList:n.array},v.defaultProps={rating:0,ratingList:[]},v.displayName="ReviewRatingData",e.exports=v},1455:function(e,t,i){Object.defineProperty(t,"__esModule",{value:!0});const n=i(2);t.default=n.defineMessages({dataLabel:{id:"ReviewRatingData.DataLabel",defaultMessage:"Rating:",description:"Label for rating"},buttonLabel:{id:"ReviewRatingData.ButtonLabel",defaultMessage:"Open rating explainer",description:"Label for rating explainer button"}})},1456:function(e,t,i){const n=i(3).default,{BaseText:o}=i(10),{calculateSpacing:a,getTypographyStyles:r,getColorToken:l}=i(4),s=n.div.withConfig({displayName:"ReviewRatingDataWrapper"})`
  display: flex;
  position: relative;
  flex-direction: row;
  align-items: center;
  margin-top: ${a(2)};
  width: 100%;
`,d=n(o).withConfig({displayName:"ReviewRatingDataLabel"})`
  margin-right: ${a(1)};
`;d.defaultProps={as:"span",typeIdentity:"typography.definitions.consumptionEditorial.subhed-aux-primary"};const c=n(o).withConfig({displayName:"ReviewRatingDataValue"})`
  ${({theme:e})=>r(e,"typography.definitions.consumptionEditorial.subhed-aux-primary")}
  color: ${({theme:e})=>l(e,"colors.consumption.body.standard.subhed")};
`,p=n.div.withConfig({displayName:"ReviewRatingDataExplainer"})`
  margin-left: ${a(.5)};

  .review-rating-data__info-button,
  .review-ratingdata__close-button {
    margin: 0;
    border: 0;
    background-color: ${({theme:e})=>l(e,"colors.interactive.base.white")};
    fill: ${({theme:e})=>l(e,"colors.consumption.body.standard.accent")};
    padding: 0;

    &:hover,
    &:focus {
      border: 0;
      background: none;
    }
  }

  .icon {
    padding-right: 5px;
    width: 24px;
    height: 24px;
  }

  .review-rating-data__info-button {
    .button__icon-container,
    .icon {
      padding-right: 0;
      width: 18px;
      height: 18px;
    }
  }

  .icon.icon-close {
    width: 35px;
    height: 35px;
  }
`,g=n.div.withConfig({displayName:"ReviewRatingDataExplainerModal"})`
  padding: ${a(2)} 0 ${a(2)}
    ${a(2)};
`,u=n.div.withConfig({displayName:"ReviewRatingDataExplainerModalList"})`
  display: flex;
  letter-spacing: 0;
`,m=n.span.withConfig({displayName:"ReviewRatingDataExplainerModalListRating"})`
  flex: 1;
  ${({theme:e})=>r(e,"typography.definitions.consumptionEditorial.citation")}
  font-weight: bold;
`,h=n.span.withConfig({displayName:"ReviewRatingDataExplainerModalListDescribe"})`
  flex: 9;
  ${({theme:e})=>r(e,"typography.definitions.consumptionEditorial.citation")}
`;e.exports={ReviewRatingDataLabel:d,ReviewRatingDataWrapper:s,ReviewRatingDataValue:c,ReviewRatingDataExplainer:p,ReviewRatingDataExplainerModal:g,ReviewRatingDataExplainerModalList:u,ReviewRatingDataExplainerModalListRating:m,ReviewRatingDataExplainerModalListDescribe:h}},1457:function(e,t,i){const{default:n}=i(3),{BREAKPOINTS:o}=i(6),{calculateSpacing:a,getColorToken:r,getTypographyStyles:l}=i(4),s=i(11),{SpanWrapper:d}=i(85),c=i(34),p=n.div.withConfig({displayName:"SponsorContentContainer"})`
  margin-top: ${a(3)};
  text-align: center;
  ${d} {
    display: inline-flex;
    margin-top: 0;
    margin-bottom: 0;
    width: 66px;
  }
`,g=n(s).withConfig({displayName:"SponsorImage"})`
  margin-right: auto;
  margin-left: auto;

  img {
    border: 1px solid;
    border-radius: 50%;
    ${({containerTheme:e,theme:t})=>((e,t)=>"inverted"===e||"special"===e?`\n      border-color: ${r(t,"colors.consumption.lead.inverted.divider")};\n    `:`\n    border-color: ${r(t,"colors.consumption.lead.standard.divider")};\n  `)(e,t)}
    width: 64px;
    height: 64px;
  }
`,u=n.div.withConfig({displayName:"SponsoredContent"})`
  ${({theme:e})=>l(e,"typography.definitions.globalEditorial.syndication")}

  display: block;
  margin: ${a(2,"px")} 0 ${a(2,"px")};
  ${({containerTheme:e,theme:t})=>((e,t)=>"inverted"===e||"special"===e?`\n      color: ${r(t,"colors.consumption.lead.inverted.syndication")};\n    `:`\n    color: ${r(t,"colors.consumption.lead.standard.syndication")};\n  `)(e,t)}

  @media (min-width: 0) and (max-width: ${o.md}) {
    margin: ${a(1.5,"px")} 0 ${a(1.5,"px")};
  }
`,m=n(c).withConfig({displayName:"SponsoredContentCampaignLink"})`
  text-decoration: none;
`;e.exports={SponsorContentContainer:p,SponsorImage:g,SponsoredContent:u,SponsoredContentCampaignLink:m}},1458:function(e,t,i){const n=i(1459);e.exports=n},1459:function(e,t,i){const n=i(0),o=i(1),a=i(67),r=i(46),l=i(165),s=i(37),d=i(11),c=i(178),p=i(91),g=i(21),u=i(501),m=i(1460),h=i(128),{useNativeShare:y}=i(93),{TextOverlayLogo:b,TextOverlayLogoLink:f,TextOverlayLogoImage:C,TextOverlayWrapper:v,ImageWrapper:w,Content:S,ContentAlign:k,Hed:x,DekWrapper:$,Dek:T,Figure:E,ContentDivider:N,ContributorImage:B,Accreditation:L,PublishDate:I,DekAndCaption:A,ContentGrid:P}=i(504),R=i(244),D=i(503),M=({background:e,bylineVariation:t,contentAlign:i,contentHeaderCategories:o,contributorImage:c,contributors:p,dangerousDek:M,dangerousHed:O,hasNativeShareButton:H,hideContributors:_,hideDangerousDek:G,hideLedeCaption:W,hidePublishDate:j,hideShareButtons:F,hideRubric:V,isLiveStoryType:U,isStoryLive:z,lede:q,ledeCaption:K,logoImage:Z,logoBaseUrl:X,numberOfLinesToClamp:Y,publishDate:J,rubric:Q,shouldUseCutomColorLiveIndicator:ee,showContentDivider:te,showContributorImage:ie,showLogo:ne,showTextOverlayDek:oe,showSponsorBlock:ae,socialDescription:re,socialMedia:le,socialTitle:se,sponsoredContentHeaderProps:de,sponsorByline:ce,theme:pe})=>{const{showNativeShareButton:ge,pageUrl:ue}=y(H),me="inverted"===pe,he=q&&!W&&(q.caption&&q.caption.trim()||q.credit&&q.credit.trim()||K);return n.createElement(v,{"data-testid":"ContentHeader"},n.createElement(w,{background:e,contentAlign:i},n.createElement(m,{lede:q}),ne&&Z?n.createElement(b,null,n.createElement(f,{href:X},n.createElement(C,Object.assign({},Z)))):null,n.createElement(P,{contentAlign:i},n.createElement(S,null,!V&&(U&&z?n.createElement(k,{contentAlign:i},n.createElement(h,{hasBackground:!0,isDiscovery:!1,shouldEnableAnimation:!0,shouldUseCutomColorLiveIndicator:ee})):Q?n.createElement(k,{contentAlign:i,"data-testid":"ContentHeaderRubric"},n.createElement(r.Inverted,Object.assign({},Q))):null),n.createElement(k,{contentAlign:i},n.createElement(u,Object.assign({},o))),n.createElement(x,{dangerouslySetInnerHTML:{__html:O},"data-testid":"ContentHeaderHed",contentAlign:i}),!G&&M&&oe&&n.createElement(T,{dangerouslySetInnerHTML:{__html:M},contentAlign:i,"data-testid":"ContentHeaderDek",isInverted:me}),te&&n.createElement(N,{contentAlign:i}),U&&ae?n.createElement(D,{sponsorByline:ce,sponsoredContentHeaderProps:Object.assign({},de)}):null,ie&&c&&n.createElement(B,{contentAlign:i},n.createElement(d,Object.assign({},c))),n.createElement(L,{contentAlign:i},p&&!_&&n.createElement(s,{contributors:p,bylineVariation:t,contentAlign:i,isCompact:!1}),!j&&n.createElement(I,{"data-testid":"ContentHeaderPublishDate",contentAlign:i},J)),!F&&(ge?n.createElement(k,{contentAlign:i,bottomSpacing:4},n.createElement(l,{hasDarkBackground:!0,shareData:{url:ue,title:se,text:re},theme:"inverted"})):le&&n.createElement(k,{contentAlign:i,bottomSpacing:4},n.createElement(g.Footer,Object.assign({},le))))))),(he||M&&!G)&&n.createElement(A,{isInverted:me},he&&n.createElement(E,{contentAlign:i},n.createElement(a,{dangerousCaptionText:q.caption,dangerousCredit:U?q.credit||K:q.credit,topSpacing:0,theme:pe})),!G&&M&&!oe&&n.createElement($,null,n.createElement(R,{isCollapsible:!0,lines:Y},n.createElement(T,{dangerouslySetInnerHTML:{__html:M},contentAlign:i,"data-testid":"ContentHeaderDek",isInverted:me})))))};M.defaultProps={background:"gradient",bylineVariation:"Inverted",contentAlign:"center",hideContributors:!1,hideDangerousDek:!1,hideLedeCaption:!1,isLiveStoryType:!1,isStoryLive:!1,logoBaseUrl:"/",numberOfLinesToClamp:2,shouldUseCutomColorLiveIndicator:!1,showContentDivider:!1,showContributorImage:!0,showSponsorBlock:!1,showTextOverlayDek:!1},M.propTypes={background:o.oneOf(["gradient","solid"]),bylineVariation:o.string,contentAlign:o.oneOf(["center","left"]),contentHeaderCategories:o.shape({title:o.string,tags:o.array}),contributorImage:o.shape(d.propTypes),contributors:o.shape(s.propTypes.contributors),dangerousDek:o.string,dangerousHed:o.string,hasNativeShareButton:o.bool,hideContributors:o.bool,hideDangerousDek:o.bool,hideLedeCaption:o.bool,hidePublishDate:o.bool,hideRubric:o.bool,hideShareButtons:o.bool,isLiveStoryType:o.bool,isStoryLive:o.bool,lede:o.oneOfType([o.shape(d.propTypes),o.shape(c.propTypes),o.shape(p.propTypes)]),ledeCaption:o.string,logoBaseUrl:o.string,logoImage:o.shape(d.propTypes),numberOfLinesToClamp:o.number,publishDate:o.string,rubric:o.shape(r.propTypes),shouldUseCutomColorLiveIndicator:o.bool,showContentDivider:o.bool,showContributorImage:o.bool,showLogo:o.bool,showSponsorBlock:o.bool,showTextOverlayDek:o.bool,socialDescription:o.string,socialMedia:o.shape(g.propTypes),socialTitle:o.string,sponsorByline:o.string,sponsoredContentHeaderProps:o.shape({campaignUrl:o.string,sponsorLogo:o.shape(d.propTypes),sponsorName:o.string}),theme:o.oneOf(["standard","inverted","special"])},M.displayName="TextOverlay",e.exports=M},1460:function(e,t,i){const n=i(0),o=i(1),a=i(11),r=i(178),l=i(91),{transformLegacySources:s}=i(93),{Image:d}=i(504),c=({lede:e})=>{if(!e||0===Object.keys(e).length)return null;const t="cnevideo"===e.modelName,i="gallery"===e.modelName,o=s(e);return n.createElement(d,null,!t&&!i&&n.createElement(a,Object.assign({},o)),t&&e.scriptEmbedUrl&&n.createElement(l,{shouldAutoplay:!0,scriptUrl:e.scriptEmbedUrl}),i&&n.createElement(r,Object.assign({},e,{showNoAdsFromParent:!0})))};c.defaultProps={lede:null},c.propTypes={lede:o.oneOfType([o.shape(a.propTypes),o.shape(r.propTypes),o.shape(l.propTypes)])},c.displayName="ImageBlock",e.exports=c},1461:function(e,t,i){e.exports=i(1462)},1462:function(e,t,i){const n=i(8),o=i(1),a=i(0),{useIntl:r}=i(2),l=i(44),s=i(1463).default,{mapSourcesToSegmentedSources:d}=i(135),c=i(1464),p=i(11),g=i(46),u=i(1433),m=i(1465),h=i(37),y=i(136),b=i(21),f=i(109),C=i(502),{formatInfoSliceItems:v}=i(1469),w=i(165),{useNativeShare:S}=i(93),{getThemedBylineVariation:k}=i(253),{trackComponent:x}=i(5),{SplitScreenContentHeaderArtist:$,SplitScreenContentHeaderArtistSlash:T,SplitScreenContentHeaderArtistWrapper:E,SplitScreenContentHeaderByline:N,SplitScreenContentHeaderCaption:B,SplitScreenContentHeaderContributorImage:L,SplitScreenContentHeaderDek:I,SplitScreenContentHeaderDekDown:A,SplitScreenContentHeaderDivider:P,SplitScreenContentHeaderHed:R,SplitScreenContentHeaderMain:D,SplitScreenContentHeaderInfoSlice:M,SplitScreenContentHeaderForMusicReview:O,SplitScreenContentHeaderNativeShareButton:H,SplitScreenContentHeaderPublishDate:_,SplitScreenContentHeaderRating:G,SplitScreenContentHeaderRubric:W,SplitScreenContentHeaderRubricIssueDate:j,SplitScreenContentHeaderSignageRubric:F,SplitScreenContentHeaderSocialShare:V,SplitScreenContentHeaderTitleBlock:U,SplitScreenContentHeaderWrapper:z,SplitScreenContentHeaderScoreBox:q,SplitScreenContentHeaderLeadWrapper:K,SplitScreenContentHeaderArtistLink:Z,SplitScreenContentHeaderGrid:X,SplitScreenContentHeaderPersistentAside:Y,SplitScreenContentHeaderReleaseYear:J}=i(252),{SplitScreenContentHeaderSocialIcons:Q}=i(446),{doHideBookmark:ee,doDisplayBookmark:te}=i(442),ie=({contributors:e,contributorsPosition:t,hasInvertedBylineLink:i,hideContributors:n,hideIssueDate:o,hidePublishDate:r,issueDate:l,issueDatePostfix:s,issueLink:d,publishDate:c,publishDatePosition:p,rubric:u,rubricVariation:m,hideRubric:h,themedBylineVariation:y})=>{const b=g[m]||g,f=g.Item,C=e&&0!==Object.keys(e).length;return a.createElement("div",{"data-testid":"ContentHeaderRubric"},C&&!n&&"top"===t&&a.createElement(N,{contributors:e,bylineVariation:y,isCompact:!1,contributorsPosition:t,hasInvertedBylineLink:i}),u&&!h&&a.createElement(W,Object.assign({as:b},u)),!o&&l&&a.createElement(j,{as:f,name:s?`${l}${s}`:l,url:d}),!r&&c&&"top"===p&&a.createElement(_,{"data-testid":"ContentHeaderPublishDate"},c))};ie.propTypes={contributors:o.object,contributorsPosition:o.oneOf(["top","bottom"]),hasInvertedBylineLink:o.bool,hideContributors:o.bool,hideIssueDate:o.bool,hidePublishDate:o.bool,hideRubric:o.bool,issueDate:o.string,issueDatePostfix:o.string,issueLink:o.string,publishDate:o.string,publishDatePosition:o.oneOf(["top","bottom"]),rubric:o.shape(g.propTypes),rubricVariation:o.string,themedBylineVariation:o.string};const ne=({signage:e,shouldDisplaySignage:t})=>e&&t?a.createElement("div",{"data-testid":"ContentHeaderRubricSignage"},a.createElement(F,{name:e})):null;ne.propTypes={shouldDisplaySignage:o.bool,signage:o.string};const oe=(e,t,i,n,o,r)=>o&&a.createElement(V,Object.assign({},o,{hasSocialShare:!0,hasStickySocialIcons:e,hideSocialIcons:t,hideSocialIconsOnMobile:i,socialIconsToHide:n,socialMediaPositionInMobile:r})),ae=(e,t)=>((null==t?void 0:t.caption)||(null==t?void 0:t.credit))&&!e&&a.createElement(X,null,a.createElement(B,{dangerousCaptionText:t.caption,dangerousCredit:t.credit})),{useRef:re,useEffect:le}=a,se=({artists:e,brandSlug:t,captionPosition:o,className:p,contentAlign:g,contributorImage:h,dangerousHed:y,dangerousDek:b,hasContributorImageBackground:f,hasInvertedBylineLink:B,hasMargin:W,hasStickySocialIcons:j,hasNativeShareButton:F,hideContributorTitle:V,hideContributors:se,hideDangerousDek:de,hideIssueDate:ce,hidePublishDate:pe,hideRubric:ge,socialIconsToHide:ue,hideSocialIcons:me,hideSocialIconsOnMobile:he,hideCaption:ye,imageAlignment:be,infoSliceFields:fe,isInset:Ce,isMusicReview:ve,isTextRight:we,isTrackReview:Se,issueDate:ke,issueDatePostfix:xe,issueLink:$e,isSplitScreenArtistLarge:Te,rubric:Ee,rubricVariation:Ne,contributors:Be,contributorsPosition:Le,bylineVariation:Ie,persistentAsideAlign:Ae,publishDate:Pe,publishDatePosition:Re,lede:De,ledeContentAlign:Me,musicRating:Oe,shouldEnableNativeShareOnDesktop:He,shouldFitToViewport:_e,showBookmarked:Ge,shouldHeaderFitToViewport:We,showContentDivider:je,showContributorImage:Fe,showHeaderDivider:Ve,socialDescription:Ue,socialMedia:ze,socialMediaPositionInMobile:qe,socialTitle:Ke,stickySocialAnchorBottom:Ze,stickySocialAnchorTop:Xe,theme:Ye,rating:Je,signage:Qe,shouldDisplaySignage:et,showReviewLink:tt,textAlign:it})=>{a.useEffect(()=>{x("SplitScreenContentHeader")},[]);const nt=k({bylineVariation:Ie,theme:Ye}),{showNativeShareButton:ot,pageUrl:at}=S(F,He),{score:rt,isBestNewMusic:lt,isBestNewReissue:st}=Oe,dt=v(fe),ct=null==fe?void 0:fe.releaseYear,pt=(e=>{if(!e)return;const t=Object.assign({},e);return new Set(["photo","cartoon"]).has(e.contentType)&&!e.segmentedSources&&e.sources&&(t.segmentedSources=d(e.sources)),t})(De),gt="cnevideo"===(null==De?void 0:De.modelName),ut=!De||gt,{rating:mt,count:ht}=Je||{},{BookmarkIcon:yt}=i(156),bt=ve?O:D,{formatMessage:ft}=r(),Ct={},vt=re();"belowImage"===o&&(Ct.captionCredit=ae(ye,De)),"inLeadWrapperBelowImg"===qe&&(Ct.socialMedia=oe(j,me,he,ue,ze,qe));const wt=()=>{(e=>{const t=e.current&&e.current.offsetTop,i=window.pageYOffset;return Math.abs(i)>Math.abs(t)})(vt)?te():ee()};return le(()=>{const e=l(wt,100);return window.addEventListener("scroll",e,{passive:!0}),()=>{window.removeEventListener("scroll",e)}}),a.createElement(z,{className:n("content-header",{[p]:p}),contentHeaderTheme:Ye,isFullWidth:ut,isTextRight:we,isInset:Ce,imageAlignment:be,ledeContentAlign:Me,shouldFitToViewport:!_e,isMusicReview:ve,"data-testid":"SplitScreenContentHeaderWrapper",showHeaderDivider:Ve,socialMediaPositionInMobile:qe,shouldHeaderFitToViewport:We,captionPosition:o,hidePublishDate:pe,mediaContentType:(null==pt?void 0:pt.contentType)||"",hasInvertedBylineLink:B,hasMargin:W},a.createElement(bt,{shouldFitToViewport:!_e},a.createElement(U,{contentAlign:g,textAlign:it},a.createElement(ne,{signage:Qe,shouldDisplaySignage:et}),a.createElement(ie,Object.assign({},{contributors:Be,contributorsPosition:Le,rubric:Ee,rubricVariation:Ne,hideContributors:se,hideIssueDate:ce,hidePublishDate:pe,issueDate:ke,issueDatePostfix:xe,issueLink:$e,publishDate:Pe,publishDatePosition:Re,hideRubric:ge,hasInvertedBylineLink:B,themedBylineVariation:nt})),a.createElement(R,{dangerouslySetInnerHTML:{__html:y},"data-testid":"ContentHeaderHed",isMusicReview:ve}),je&&a.createElement(P,{ledeContentAlign:Me}),e&&ve?a.createElement(E,{isMusicReview:ve},0===e.length&&a.createElement($,{isSplitScreenArtistLarge:Te},ft(s.variousArtists)),e.map((t,i)=>a.createElement(a.Fragment,{key:i},a.createElement(Z,{key:i,href:"/".concat(t.uri)},a.createElement($,{dangerouslySetInnerHTML:{__html:t.name},isSplitScreenArtistLarge:Te})),!(i===e.length-1)&&a.createElement(T,{dangerouslySetInnerHTML:{__html:" / "},isSplitScreenArtistLarge:Te})))):!de&&b&&a.createElement(I,{dangerouslySetInnerHTML:{__html:b}}),Fe&&h&&a.createElement(L,Object.assign({},h,{sizes:"66px",hasContributorImageBackground:f})),Be&&!se&&"bottom"===Le&&a.createElement(N,{contributors:Be,bylineVariation:nt,isCompact:!1,hasInvertedBylineLink:B}),!pe&&"bottom"===Re&&a.createElement(_,{"data-testid":"ContentHeaderPublishDate"},Pe),(ve||Se)&&ct&&a.createElement(J,{"data-testid":"SplitScreenContentHeaderReleaseYear"},ct),Ge&&a.createElement("div",{ref:vt},a.createElement(yt,{bookmarkTrackingType:"header",link:{label:"Save story",url:"#",network:"bookmark",behavior:"bookmark"},theme:"standard",type:"thin",isUrlBookmark:!0,isBookmarkButton:!0})),!!mt&&!!ht&&a.createElement(G,{averageRatingCount:Math.round(10*mt)/10,brandSlug:t,hasBorderTop:!0,link:tt&&{label:ft(s.ratingLinkLabel),onClick:e=>{e.preventDefault();const t=document.getElementById("reviews"),{top:i}=C(t);null==t||t.focus(),window.scrollTo(0,i-56)},url:"#reviews"},totalRatingCount:ht}),ot?a.createElement(H,null,a.createElement(w,{shareData:{url:at,title:Ke,text:Ue}})):oe(j,me,he,ue,ze,qe)),a.createElement(K,{isMusicReview:ve},a.createElement(c,Object.assign({lede:pt,isCNEVideo:gt},Ct)),ve&&a.createElement(q,null,a.createElement(m,{rating:rt,isBestNewMusic:lt,isBestNewReissue:st,isTrackReview:Se})))),"belowHeader"===o&&ae(ye,De),Be&&se&&a.createElement(N,{contributors:Be,bylineVariation:V?"Item":Ie,isCompact:!1,isMusicReview:ve}),dt.length>0&&a.createElement(X,null,a.createElement(M,null,a.createElement(u,{items:dt,isMusicReview:ve}))),!de&&ve&&b&&a.createElement(X,null,a.createElement(A,{dangerouslySetInnerHTML:{__html:b}})),!ot&&j&&ze&&a.createElement(Y,{align:Ae,attributes:{shouldFadeOnMove:!0},anchorBottom:Ze,anchorTop:Xe,fullWidthSelector:".container--full, .full-bleed-ad, .callout--feature-large"},a.createElement(Q,Object.assign({},ze,{bookmarkTrackingType:"sticky",className:"social-icons--share"}))))};se.propTypes={artists:o.array,brandSlug:o.string,bylineVariation:o.string,captionPosition:o.oneOf(["belowHeader","belowImage"]),className:o.string,contentAlign:o.oneOf(["center","left"]),contributorImage:o.shape(p.propTypes),contributors:o.shape(h.propTypes.contributors),contributorsPosition:o.oneOf(["top","bottom"]),dangerousDek:o.string,dangerousHed:o.string.isRequired,hasContributorImageBackground:o.bool,hasInvertedBylineLink:o.bool,hasMargin:o.bool,hasNativeShareButton:o.bool,hasStickySocialIcons:o.bool,hideCaption:o.bool,hideContributors:o.bool,hideContributorTitle:o.bool,hideDangerousDek:o.bool,hideIssueDate:o.bool,hidePublishDate:o.bool,hideRubric:o.bool,hideSocialIcons:o.bool,hideSocialIconsOnMobile:o.bool,imageAlignment:o.oneOf(["center","top","left","right","bottom"]),infoSliceFields:o.object,isInset:o.bool,isMusicReview:o.bool,isSplitScreenArtistLarge:o.bool,issueDate:o.string,issueDatePostfix:o.string,issueLink:o.string,isTextRight:o.bool,isTrackReview:o.bool,lede:o.oneOfType([o.shape(p.propTypes)]),ledeContentAlign:o.oneOf(["left","center"]),musicRating:o.object,persistentAsideAlign:o.oneOf(["left","left-lead-asset"]),publishDate:o.string.isRequired,publishDatePosition:o.oneOf(["top","bottom"]),rating:o.shape(f.propTypes),rubric:o.shape(g.propTypes),rubricVariation:o.string,shouldDisplaySignage:o.bool,shouldEnableNativeShareOnDesktop:o.bool,shouldFitToViewport:o.bool,shouldHeaderFitToViewport:o.bool,showBookmarked:o.bool,showContentDivider:o.bool,showContributorImage:o.bool,showHeaderDivider:o.bool,showReviewLink:o.bool,signage:o.string,socialDescription:o.string,socialIconsToHide:o.array,socialMedia:o.shape(b.propTypes),socialMediaPositionInMobile:o.oneOf(["inLeadWrapperBelowImg","inTitleBlock"]),socialTitle:o.string,stickySocialAnchorBottom:y.propTypes.anchorBottom,stickySocialAnchorTop:y.propTypes.anchorTop,textAlign:o.oneOf(["left","center"]),theme:o.oneOf(["standard","inverted","special"])},se.defaultProps={brandSlug:"",captionPosition:"belowHeader",contentAlign:"center",contributorsPosition:"bottom",hasContributorImageBackground:!1,hasInvertedBylineLink:!1,hasMargin:!1,hasStickySocialIcons:!0,hideCaption:!1,hideContributorTitle:!1,hideContributors:!1,hideDangerousDek:!1,hidePublishDate:!1,hideRubric:!1,hideSocialIcons:!1,hideSocialIconsOnMobile:!1,imageAlignment:"center",isInset:!1,isTextRight:!1,issueDatePostfix:" Issue",ledeContentAlign:"left",musicRating:{score:null},persistentAsideAlign:"left",publishDatePosition:"bottom",shouldEnableNativeShareOnDesktop:!1,shouldFitToViewport:!0,shouldHeaderFitToViewport:!1,showBookmarked:!1,showContentDivider:!1,showContributorImage:!0,showHeaderDivider:!0,showReviewLink:!0,socialIconsToHide:[],socialMediaPositionInMobile:"inTitleBlock",stickySocialAnchorBottom:{selector:".page",edge:"bottom"},stickySocialAnchorTop:{selector:"[data-testid='SplitScreenContentHeaderWrapper']",edge:"bottom"},textAlign:"center",theme:"standard"},se.displayName="SplitScreenContentHeader",e.exports=se},1463:function(e,t,i){Object.defineProperty(t,"__esModule",{value:!0});const n=i(2);t.default=n.defineMessages({ratingLinkLabel:{id:"SplitScreenContentHeader.RatingLinkLabel",defaultMessage:"Read Reviews",description:"SplitScreenContentHeader component Rating Link Label"},variousArtists:{id:"SplitScreenContentHeader.VariousArtists",defaultMessage:"Various Artists",description:"SplitScreenContentHeader component various artists text"}})},1464:function(e,t,i){const n=i(0),o=i(1),a=i(91),{SplitScreenContentHeaderLede:r,SplitScreenContentHeaderLedeBlock:l}=i(252),s=({captionCredit:e,className:t,isCNEVideo:i,lede:o,socialMedia:s})=>o?i?o.scriptEmbedUrl?n.createElement(l,{"data-testid":"ContentHeaderLeadAsset",className:t},n.createElement(a,{hasMargins:!1,shouldAutoplay:!0,scriptUrl:o.scriptEmbedUrl})):null:n.createElement(l,{"data-testid":"ContentHeaderLeadAsset",className:t},n.createElement(r,Object.assign({},o)),e,s):null;s.propTypes={captionCredit:o.object,className:o.string,isCNEVideo:o.bool,lede:o.object,socialMedia:o.object},e.exports=s},1465:function(e,t,i){e.exports=i(1466)},1466:function(e,t,i){const n=i(0),{useIntl:o}=i(2),a=i(1),{BestNewMusicArrows:r}=i(1467),l=i(1468).default,s=i(76),{trackComponent:d}=i(5),{BestNewMusicText:c,Rating:p,ScoreBoxWrapper:g,ScoreCircle:u}=i(505),m=({rating:e,isBestNewMusic:t,isBestNewReissue:i,palette:a,isTrackReview:m})=>{n.useEffect(()=>{d("ScoreBox")},[]);const{formatMessage:h}=o(),y=m&&t;if(m&&!t)return null;const b=e<10?Number(e).toFixed(1):e,f=t||i,C=t&&i;return n.createElement(s,{palette:a},n.createElement(g,null,f&&n.createElement(r,{isBestBoth:C,isBest:f}),!y&&n.createElement(u,{isBest:f,isBestBoth:C},n.createElement(p,{isBest:f,isBestBoth:C},b)),f&&n.createElement(c,{isBestBoth:C,isBest:f},!y&&t&&n.createElement("div",null," ",h(l.BestNewMusic)," "),!y&&i&&n.createElement("div",null," ",h(l.BestNewReissue)),y&&n.createElement("div",null," ",h(l.BestNewTrack)))))};m.propTypes={isBestNewMusic:a.bool,isBestNewReissue:a.bool,isTrackReview:a.bool,palette:a.oneOf(["standard","inverted"]),rating:a.number},m.defaultProps={isBestNewMusic:!1,isBestNewReissue:!1,palette:"standard"},e.exports=m},1467:function(e,t,i){Object.defineProperty(t,"__esModule",{value:!0}),t.BestNewMusicArrows=void 0;const n=i(0),o=i(1),{SvgWrapper:a,SvgStyle:r}=i(505);t.BestNewMusicArrows=e=>n.createElement(a,{theme:e.theme,isBestBoth:e.isBestBoth,isBest:e.isBest},n.createElement(r,{x:"0px",y:"0px",viewBox:"0 0 80 40"},n.createElement("g",null,n.createElement("polyline",{points:"25.4,14.7 33.9,14.7 33.9,39.8 46.3,39.8 46.3,14.7 54.8,14.7 40.1,0 25.4,14.7   "}),n.createElement("polyline",{points:"50.6,40 80,40 65.2,25.4 50.6,40    "}),n.createElement("polyline",{points:"0,40 29.4,40 14.7,25.4 0,40    "})))),t.BestNewMusicArrows.propTypes={isBest:o.bool,isBestBoth:o.bool,theme:o.string}},1468:function(e,t,i){Object.defineProperty(t,"__esModule",{value:!0});const n=i(2);t.default=n.defineMessages({BestNewMusic:{id:"ScoreBox.BestNewMusic",defaultMessage:"Best New Music",description:"Best New Music"},BestNewReissue:{id:"ScoreBox.BestNewReissue",defaultMessage:"Best New Reissue",description:"Best New Reissue"},BestNewTrack:{id:"ScoreBox.BestNewTrack",defaultMessage:"Best New Track",description:"Best New Track"}})},1469:function(e,t){const i={genre:"Genre:",label:"Label:",reviewDate:"Reviewed:"};e.exports={formatInfoSliceItems:e=>{if(!e)return[];const t=[];return Object.keys(e).forEach(n=>{e[n]&&e[n].length&&i[n]&&t.push({key:i[n],value:e[n]})}),t}}},1481:function(e,t,i){const{asConfiguredComponent:n}=i(9),{asThemedComponent:o}=i(30),a=i(1430);e.exports=o(n(a,"ContentHeader"))},1486:function(e,t,i){e.exports=i(1497)},1497:function(e,t,i){const n=i(1),o=i(0),{connect:a}=i(20),{useIntl:r}=i(2),l=i(1498).default,s=i(27).default,d=i(32),c=i(11),{googleAnalytics:p}=i(13),{asConfiguredComponent:g}=i(9),{doCloseModal:u}=i(160),{trackComponent:m}=i(5),{SignInModalBaseWrapper:h,SignInModalCloseButton:y,SignInModalButtons:b,SignInModalDek:f,SignInModalHed:C,SignInModalEmail:v,SignInModalSignInSection:w,SignInModalSpotIllustration:S,SignInModalSignInLink:k,SignInModalHedSpanTag:x}=i(1499),$=({analyticsType:e,authSource:t,brandIdentityAssets:i,className:n,closeButtonCallback:a,dangerousDek:g,dangerousHed:$,dangerousHedSpanTag:T,hasLargeMargins:E,hasRoundedCornersButtons:N,isVisible:B,redirectURL:L,source:I,type:A})=>{o.useEffect(()=>{m("SignInModal")},[]);const P=t=>p.emitGoogleTrackingEvent(t,{signInModalType:e});B&&P("signin-modal-impression");let R=`${s.oidcAuth}?redirectURL=${encodeURIComponent(L)}&skipAccountLink=true&authSource=${t}`;I&&(R=`${R}&source=${I}`);const D=i.signInModalAsset[A]||i.signInModalAsset.default,{formatMessage:M}=r();return o.createElement(h,{className:n,hasLargeMargins:E,closeTimeoutMS:400,isOpen:B},o.createElement(y,{isIconButton:!0,ariaLabel:M(l.closeButtonAriaLabel),label:M(l.closeButtonLabel),onClickHandler:()=>{u(),P("signin-modal-close"),a&&a()},role:"button",ButtonIcon:d}),o.createElement(C,null,$||M(l.hed),o.createElement(x,null,T||M(l.hedSpanTag))),o.createElement(S,null,o.createElement(c,Object.assign({},D))),o.createElement(f,{dangerouslySetInnerHTML:{__html:g}}),o.createElement(b,{hasRoundedCornersButtons:N},o.createElement(v,{label:M(l.oidcSignUpButtonLabel),href:R,inputKind:"link",rel:"link",iconPosition:"before",onClickHandler:()=>P("signin-modal-oidc-sign-in-click"),"data-testid":"SignInModalEmail"})),o.createElement(w,null,M(l.oidcSignInText)," ",o.createElement(k,{href:R,onClick:()=>P("signin-modal-oidc-sign-in-click")},M(l.oidcSignInLinkText)),"voting"===A&&o.createElement("div",null,"  »")))};$.displayName="SignInModal",$.defaultProps={authSource:"sign-in-modal",hasLargeMargins:!1,redirectURL:"/",type:"default"},$.propTypes={analyticsType:n.string.isRequired,authSource:n.string,brandIdentityAssets:n.shape({signInModalAsset:n.shape({default:n.object,crosswords:n.object,voting:n.object})}).isRequired,className:n.string,closeButtonCallback:n.func,dangerousDek:n.string,dangerousHed:n.string,dangerousHedSpanTag:n.string,hasLargeMargins:n.bool,hasRoundedCornersButtons:n.bool,isVisible:n.bool,redirectURL:n.string,source:n.string,type:n.oneOf(["default","crosswords","voting"])};const T=g($,"SignInModal");e.exports=a(e=>{const{signInModalConfig:t,brandIdentityAssets:i}=e;return Object.assign({brandIdentityAssets:i},t)})(T)},1498:function(e,t,i){Object.defineProperty(t,"__esModule",{value:!0});const n=i(2);t.default=n.defineMessages({oidcSignUpButtonLabel:{id:"SignInModal.OidcSignUpButtonLabel",defaultMessage:"Create account",description:"SignInModal component OIDC SignUp button label"},closeButtonLabel:{id:"SignInModal.CloseButtonLabel",defaultMessage:"Close Sign In Modal",description:"SignInModal component close button label"},closeButtonAriaLabel:{id:"SignInModal.CloseButtonAriaLabel",defaultMessage:"Close Sign In Modal",description:"SignInModal component close button aria label"},hed:{id:"SignInModal.Hed",defaultMessage:"Save stories ",description:"SignInModal component hed text",isConfigurable:!0},hedSpanTag:{id:"SignInModal.HedSpanTag",defaultMessage:"with an account",description:"SignInModal component hed spanTag text",isConfigurable:!0},oidcSignInText:{id:"SignInModal.OidcSignInText",defaultMessage:"Already have an account?",description:"SignInModal component OIDC SignIn Text"},oidcSignInLinkText:{id:"SignInModal.OidcSignInLinkText",defaultMessage:"Sign in",description:"SignInModal component OIDC SignIn Link Text"}})},1499:function(e,t,i){var n=this&&this.__rest||function(e,t){var i={};for(var n in e)Object.prototype.hasOwnProperty.call(e,n)&&t.indexOf(n)<0&&(i[n]=e[n]);if(null!=e&&"function"==typeof Object.getOwnPropertySymbols){var o=0;for(n=Object.getOwnPropertySymbols(e);o<n.length;o++)t.indexOf(n[o])<0&&Object.prototype.propertyIsEnumerable.call(e,n[o])&&(i[n[o]]=e[n[o]])}return i};const o=i(0),a=i(154),r=i(1),{default:l,css:s}=i(3),{BaseText:d,BaseLink:c}=i(10),{calculateSpacing:p}=i(4),{BREAKPOINTS:g,ZINDEX_MAP:u}=i(6),m=i(14),{getColorToken:h,getTypographyStyles:y}=i(4),{ResponsiveImagePicture:b}=i(24),{ButtonWrapper:f}=i(29),C=l(d).withConfig({displayName:"SignInModalHed"})`
  text-align: center;
`;C.defaultProps={as:"div",colorToken:"colors.discover.body.white.heading",topSpacing:1,typeIdentity:"typography.definitions.consumptionEditorial.display-small"};const v=l.span.withConfig({displayName:"SignInModalHedSpanTag"})`
  display: block;
`,w=l.p.withConfig({displayName:"SignInModalDek"})`
  ${y("typography.definitions.consumptionEditorial.body-core")}
  text-align: center;
  color: ${h("colors.discover.body.white.description")};
  @media (max-width: ${g.md}) {
    margin-bottom: ${p(3)};
  }
`,S=l(d).withConfig({displayName:"SignInModalSpotIllustration"})`
  margin: ${p(1.5)} auto;
  text-align: center;

  ${b} {
    width: 200px;
    height: 170px;
  }
`;S.defaultProps={as:"div",typeIdentity:"typography.definitions.consumptionEditorial.body-core"};const k=l(m.Primary).withConfig({displayName:"SignInModalButtonPrimary"})`
  margin-bottom: ${p(1)};
  width: 100%;
`,x=l(m.Primary).withConfig({displayName:"SignInModalButtonPrimaryOutlined"})`
  width: 100%;
`,$=l(m.Utility).withConfig({displayName:"SignInModalCloseButton"})`
  position: absolute;
  top: ${p(1)};
  right: ${p(1)};
  padding: 0;
  fill: ${h("colors.discovery.body.light.context-tertiary")};

  &,
  &:focus,
  &:hover {
    border: 0;
    background-color: transparent;
  }
`,T=l(m.Utility).withConfig({displayName:"SignInModalEmail"})`
  margin-top: 0;
  padding: 0;
`,E=l.div.withConfig({displayName:"SignInModalButtons"})`
  display: flex;
  justify-content: center;
  margin: auto;
  padding-bottom: ${p(2)};
  width: 100%;

  ${T} {
    padding: 0;
    width: 100%;
  }

  ${({hasRoundedCornersButtons:e})=>e&&`\n    ${f} {\n      border-radius: 4px;\n    }\n  `}
`,N=s`
  &&& {
    padding: ${p(8)} ${p(5)};
    min-height: ${p(62)};
  }
  ${S} {
    margin: ${p(1.5)} auto;

    ${b} {
      width: 175px;
      height: 175px;
    }
  }

  ${w} {
    margin-bottom: ${p(2)};
  }

  ${E} {
    margin-bottom: ${p(4)};

    @media (max-width: ${g.md}) {
      padding-bottom: 0;
    }
  }
`;function B(e){var{className:t}=e,i=n(e,["className"]);const r=t+"__content",l=t+"__overlay";return o.createElement(a,Object.assign({portalClassName:t,className:r,overlayClassName:l},i))}B.propTypes={className:r.string};const L=l(B).withConfig({displayName:"SignInModalBaseWrapper"})`
  &__overlay {
    position: fixed;
    top: 0;
    right: 0;
    bottom: 0;
    left: 0;
    z-index: ${u.interstitialLayer};

    background-color: rgba(
      ${h("colors.background.black",{rgbOnly:!0})},
      0
    );

    &.ReactModal__Overlay--after-open {
      transition: background-color 750ms;
      opacity: 1;
      background-color: rgba(
        ${h("colors.background.black",{rgbOnly:!0})},
        0.4
      );
    }

    &.ReactModal__Overlay--after-close {
      transition: background-color 750ms;
      background-color: rgba(
        ${h("colors.background.black",{rgbOnly:!0})},
        0
      );
    }
  }

  &__content {
    position: relative;
    top: 50%;
    left: 50%;
    transform: translateY(-50%) translateX(-50%);
    background-color: ${h("colors.background.white")};
    padding: ${p(5)};
    width: ${p(60)};
    max-height: calc(100% - ${p(1,"px")});
    overflow-y: auto;

    @media (max-width: ${g.md}) {
      transform: translateY(-50%)
        translateX(calc(-50% - ${p(2,"px")}));

      margin: 0 ${p(2,"px")};
      padding: ${p(4)} ${p(5)}
        ${p(4)} ${p(5)};
      width: auto;
      min-width: ${p(38)};
      max-width: ${p(60)};
    }

    ${k}
    ${x}
    ${({hasLargeMargins:e})=>!0===e&&N}
  }
`,I=l(d).withConfig({displayName:"SignInModalSignInSection"})`
  display: flex;
  justify-content: center;
  width: 100%;
`;I.defaultProps={as:"div",colorToken:"colors.discovery.body.light.context-tertiary",typeIdentity:"typography.definitions.utility.assistive-text"};const A=l(c).withConfig({displayName:"SignInModalSignInLink"})`
  && {
    z-index: ${u.interstitialLayer};
    margin-left: 5px;
    text-decoration: underline;
  }
`;A.defaultProps={colorToken:"colors.interactive.base.black",typeToken:"typography.definitions.utility.assistive-text"},e.exports={SignInModalBaseWrapper:L,SignInModalCloseButton:$,SignInModalButtons:E,SignInModalDek:w,SignInModalEmail:T,SignInModalHed:C,SignInModalSignInSection:I,SignInModalSignInLink:A,SignInModalSpotIllustration:S,SignInModalHedSpanTag:v}},1519:function(e,t,i){e.exports=i(1617)},1520:function(e,t,i){const{css:n,default:o}=i(3),{calculateSpacing:a,minScreen:r,minMaxScreen:l,getColorToken:s,getTypographyStyles:d,getColorStyles:c}=i(4),{CarouselControlButton:p,CarouselList:g,CarouselWrapper:u}=i(47),m=i(19),{BREAKPOINTS:h}=i(6),y=i(78),b=o(m.EvenFour).withConfig({displayName:"FilterableSummaryListGrid"})`
  &.grid-even.grid-items-4 {
    ${r(h.md)} {
      grid-template-columns: repeat(4, minmax(0, 1fr));
    }
  }

  &.grid-even {
    ${l(0,h.md)} {
      grid-template-columns: repeat(2, minmax(0, 1fr));
      padding-right: ${a(3)};
      padding-left: ${a(3)};
    }
    padding-right: ${a(4)};
    padding-left: ${a(4)};
  }

  &.grid {
    row-gap: ${a(6)};
  }
`,f=o(m.DynamicGridItemLayout).withConfig({displayName:"FilterableSummaryListDynamicGridItemLayout"})`
  &.grid-even.grid-items-4 {
    ${r(h.md)} {
      grid-template-columns: repeat(4, minmax(0, 1fr));
    }
  }

  &.grid-even {
    ${l(0,h.md)} {
      grid-template-columns: repeat(2, minmax(0, 1fr));
      padding-right: ${a(3)};
      padding-left: ${a(3)};
    }
    padding-right: ${a(4)};
    padding-left: ${a(4)};
  }

  &.grid {
    row-gap: ${a(6)};
    ${({shouldDisplayDenseGrid:e=!0})=>!e&&"grid-auto-flow:row;"}

    ${({hasGridBottomPadding:e=!1})=>e&&`padding-bottom:${a(8)};`}
  }
`,C=o.section.withConfig({displayName:"FilterableSummaryListWrapper"})`
  ${({theme:e,hasBorderBottom:t})=>t&&`border-bottom: 1px solid ${s(e,"colors.consumption.lead.standard.divider")};`}

  ${({hasPadding:e,hasToggleGridColor:t})=>e&&!t?`padding: ${a(4)} 0 ${a(4)};`:""}
`,v=o.div.withConfig({displayName:"TitleToggleChipListWrapper"})`
  ${({shouldUseAlternativeTitleStyle:e})=>e&&n`
      .section-title {
        margin: 0;
        padding-top: 0;

        @media (max-width: ${h.md}) {
          h2 {
            text-align: center;
          }
        }
      }

      @media (max-width: ${h.md}) {
        .list-wrapper {
          overflow-y: hidden;
          overflow-x: scroll;

          &::-webkit-scrollbar {
            display: none;
          }
        }
      }

      @media (min-width: ${h.md}) {
        ${c("border-color","colors.consumption.lead.standard.context-signature")};

        display: flex;
        align-items: center;
        border-top: 2px solid;
        gap: 2rem;

        .clip-list {
          margin: 0;

          .list-wrapper {
            padding: 0;
            gap: ${a(3)};

            button {
              border-radius: 0;
              background: none;
              padding: 0.2rem 0;
              font-size: 13px;
              font-weight: bold;

              &[aria-checked='false'] {
                ${c("color","colors.interactive.base.border")};
              }

              &[aria-checked='true'] {
                ${c("color","colors.interactive.base.black")};
                border-bottom: 1px solid;
              }

              &:hover,
              &:focus {
                box-shadow: none;
              }
            }
          }
        }
      }
    `}
`,w=o.div.withConfig({displayName:"TitleWrapper"})`
  ${({hasTitleMarginTop:e})=>{const t=a(4);return`margin: ${e?t:"0"} 0 ${t} 0;`}}

  ${({hasPadding:e})=>e&&`padding-left: ${a(3)};\n    padding-right:  ${a(3)};\n   `}
  
  ${({hasTitleNoMargin:e})=>e&&"margin: 0;"}
`,S=o.div.withConfig({displayName:"ChipWrapper"})`
  ${({hasToggleGridColor:e})=>e?`padding-bottom: ${a(4)};`:`margin: ${a(4)} 0 0 0;`}
`,k=o.div.withConfig({displayName:"CarouselWrapper"})`
  ${p} {
    margin-top: 1rem;

    &:disabled {
      display: none;
    }
  }
`,x=o.section.withConfig({displayName:"EditorsPicksCarousel"})`
  display: grid;
  grid-template-columns: 100%;
  margin-top: ${a(4)};
  overflow-x: hidden;
  @media (min-width: ${h.lg}) {
    grid-column-gap: ${a(4)};
    grid-template-columns: calc(20% - ${a(4)}) 80%;
    ${u} {
      margin-top: 0;
    }
  }
  ${g} {
    margin: 0;
  }
`,$=o.div.withConfig({displayName:"EditorCard"})`
  display: grid;
  grid-template-columns: auto 1fr;
  align-items: center;
  @media (min-width: ${h.lg}) {
    grid-template-columns: 1fr;
    grid-template-rows: repeat(3, max-content);
    margin: ${a(4)} 0;
  }
`,T=o(y).withConfig({displayName:"EditorResponsiveAsset"})`
  border-radius: 50%;
  width: 80px;
  height: 80px;
`,E=o.div.withConfig({displayName:"EditorDetails"})`
  margin-left: ${a(1)};
  @media (min-width: ${h.lg}) {
    grid-column: 1;
    margin: ${a(1)} 0 0;
  }
`,N=o.div.withConfig({displayName:"EditorName"})`
  ${({theme:e})=>d(e,"typography.definitions.consumptionEditorial.subhed-aux-secondary")};
`,B=o.div.withConfig({displayName:"EditorTitle"})`
  ${({theme:e})=>d(e,"typography.definitions.globalEditorial.accreditation-feature")};
`,L=o.div.withConfig({displayName:"EditorDescription"})`
  grid-column: 1 / span 2;
  margin-top: ${a(1.5)};
  ${({theme:e})=>d(e,"typography.definitions.consumptionEditorial.description-embed")}
  @media (min-width: ${h.lg}) {
    margin-top: ${a(2.5)};
  }
`;e.exports={CarouselWrapper:k,ChipWrapper:S,EditorCard:$,EditorDescription:L,EditorDetails:E,EditorName:N,EditorResponsiveAsset:T,EditorsPicksCarousel:x,EditorTitle:B,FilterableSummaryListDynamicGridItemLayout:f,FilterableSummaryListGrid:b,FilterableSummaryListWrapper:C,TitleWrapper:w,TitleToggleChipListWrapper:v}},155:function(e,t,i){e.exports=i(158)},1563:function(e,t,i){const{default:n,createGlobalStyle:o,css:a}=i(3),{BREAKPOINTS:r}=i(6),{calculateSpacing:l,getColorStyles:s,getTypographyStyles:d,minScreen:c,minMaxScreen:p,getColorToken:g,getLinkStyles:u}=i(4),{withRecircContextProvider:m}=i(168),h=m(i(80)),{getPattern:y}=i(433),b=i(14),{ConfiguredDisclaimer:f}=i(97),{BodyWrapper:C}=i(169),v=i(493),{SectionTitleHed:w}=i(69),{LinkStackContent:S}=i(494),{RecircMostPopularWrapper:k,RecircMostPopularHeading:x}=i(485),{StickyBoxWrapper:$,StickyBoxPrimary:T}=i(468),{IframeEmbedWrapper:E}=i(269),{ResponsiveCartoonWrapper:N}=i(237),{GalleryEmbedHr:B}=i(175),{AssetEmbedAssetContainer:L}=i(106),{GridItem:I,GridWrapper:A}=i(25),P=i(136),{SocialIconsListItem:R}=i(28),{ResponsiveCartoonCaption:D}=i(237),{applyCustomBackgroundColor:M}=i(16),{ConsumerMarketingUnitThemedWrapper:O}=i(491),H=a`
  ${({pageBackgroundTheme:e})=>e?a`
        ${M(e)};
      `:a`
      ${({theme:e})=>y(e,"page-background")};
    `}
  .grid-layout__content {
    ${c(r.md)} {
      grid-column: 3 / span 8;
    }

    ${c(r.lg)} {
      grid-column: 2 / span 6;
    }

    ${p(r.sm,r.md)} {
      grid-column: 1 / -1;
    }
  }

  .grid-layout--adrail.narrow {
    .container--body-inner {
      ${c(r.md)} {
        grid-column: 1 / -1;
      }
    }

    ${k} {
      &:first-child {
        margin-top: 0;

        ${x} {
          margin-top: 0;
        }
      }
    }
  }

  .container--body {
    grid-gap: 20px;
  }

  inline-embed[name='align-right'] {
    text-align: right;
  }

  inline-embed[name='align-center'] {
    text-align: center;
  }
`,_=n(h).withConfig({displayName:"ArticlePageBase"})`
  &&& {
    ${H}
    ${({shouldHideBaseTopPadding:e})=>e&&"padding-top : 0;"}
    ${({hideNav:e,shouldPrioritizeSeriesPagination:t,hasContentHeaderLogo:i})=>!t&&i&&(e=>e?"\n    header.site-navigation {\n      margin-bottom: -7rem;\n      transform: translateY(-150%);\n      transition: all 1000ms ease;\n    }\n    ":"\n    header.site-navigation {\n      margin-bottom: -7rem;\n      transition: all 500ms ease;\n    }\n")(e)}
  }
`,G=o`
    .channel--body {
      ${H}
    }
    
    .body__container {
      grid-column: 1/ -1;
    
      ${c(r.md)} {
        grid-column: 3 / span 8;
      }
    }
  
    /* 1. required to enforce proper alignment when text may be less than a full line
       2. remove extra top spacing added by default paragraph margins */
      .article__body {
        margin-bottom: ${l(2)};
        width: 100%; /* 1 */
  
        p:first-child:not(.callout--group-item) {
          margin-top: 0; /* 2 */
        }
  
        .small {
          font-variant: small-caps;
          text-transform: lowercase;
          font-style: normal;
        }
  
        .underline {
          text-decoration: underline;
          font-style: normal;
        }

        ${D} .underline {
          font-style: inherit;
        }
      }

      .article-white-background {
        background-color: white;
        padding: 1rem;
      }
  
      .article__body > .body__inner-container > {
        ${E}, .cne-video-embed {
            &:first-child {
              margin-top: 0;
            }
          }
  
          inline-embed:first-child ${E} {
            margin-top: 0;
          }
      }

      .article__body--grid-margins {
        grid-column: 1 / -1;
      }

      /**
       1. have ad span more columns from the right panel
       to have a larger right margin
       2. have it span 3 which is intended in a 12 column grid
       */
       .grid-layout__aside {
         display: none;
     
         ${c(r.lg)} {
           display: block;
           min-width: 300px;
     
           ${$} {
             top: 90px;
           }
         }
     
         ${k} {
           &:first-child {
             margin-top: 0;
     
             ${x} {
               margin-top: 0;
             }
           }
         }
       }

        ${C} {
          ${({dividerColor:e})=>e&&`\n                .body__inner-container  > hr {\n                    background: #${e};\n                    height: 1px;\n                }\n              `}
       }

  `,W=n("div").withConfig({displayName:"PaywallInlineBarrierWithWrapperGrid"})`
  ${({adRail:e})=>!e&&`\n    > ${I} {\n      grid-column: 1 / -1;\n    }`}
`,j=n("div").withConfig({displayName:"ArticlePageLedeBackground"})`
  ${({theme:e})=>y(e,"lede-background")}
`,F=n("div").withConfig({displayName:"ArticlePageContentBackGround"})`
  ${({theme:e})=>y(e,"lede-background")}
  padding-top: ${l(2)};

  @media (min-width: 1208px) {
    padding-top: ${l(4)};
  }

  ${({togglePaddingTop:e})=>"large"===e&&`padding-top: ${l(4)};\n       ${c(r.md)}{ \n        padding-top: ${l(6)};\n       }\n      `}

  ${({togglePaddingTop:e})=>"xlarge"===e&&`${c(r.lg)} {  padding-top: ${l(8)}; }`}
  
    ${({isMobileTruncated:e})=>e&&`\n          position: relative;\n          height: 890px;\n          overflow: hidden;\n  \n          &::before {\n            display: block;\n            position: absolute;\n            bottom: 0;\n            z-index: 1;\n            background: linear-gradient(\n              0deg,\n              rgba(255, 255, 255, 1) 20%,\n              rgba(255, 255, 255, 0) 100%\n            );\n            width: 100%;\n            height: 192px;\n            content: '';\n          }\n  \n          ${c(r.md)} {\n            height: auto;\n            overflow: visible;\n  \n            &::before {\n              display: none;\n            }\n          }\n        `}

  ${({cartoonVariation:e})=>"card"===e&&a`
      ${N}::before, ${N}::after {
        border: none;
      }

      ${N} {
        ${s("background-color","colors.background.brand")};
        border: none;
        padding: ${l(3)} ${l(6)};

        ${c(r.lg)} {
          ${R} a {
            width: ${l(5)};
          }
        }
      }
    `}
`,V=n("div").withConfig({displayName:"ArticlePageChunks"})`
  .grid:last-child {
    .body__container > .body__inner-container > *:last-child {
      ${B}:last-child {
        display: none;
      }
    }
  }

  ${({horizontalRuleStyle:e})=>"thin"===e&&"\n        .body__container {\n          .callout--has-top-border {\n            border-top-width: 1px;\n          }\n\n          hr {\n            height: 1px;\n          }\n        }\n      "}

  ${({hasTopSpacing:e})=>!e&&`\n        .inset-embedded-lede {\n          @media (min-width: 0px) and (max-width: calc(${r.md} - 1px)) {\n            ${L} {\n              transform: translateX(-50%);  /* 1 */\n              margin-left: 50%;\n              width: 100vw;\n            }\n          }\n        }\n\n        ${c(r.md)} {\n          .body__container {\n            p:first-of-type {\n              margin-top: 0;\n            }\n\n            .inset-embedded-lede {\n              margin-top: 0;\n            }\n          }\n        }\n    `}

  @media print {
    ${A} {
      display: block;
    }

    ${A} > p {
      font-size: 17px;
    }
  }

  ${({pageBackgroundTheme:e})=>e&&".ad.ad--mid-content {\n      .ad-label {\n        color: #1A1A1A;\n      }\n    }"}
`,U=n(b.Utility).withConfig({displayName:"ArticlePageBodyMobileTruncatedBtn"})`
  position: absolute;
  bottom: 0;
  left: 50%;
  z-index: 2;
  margin-left: -100px;
  width: 200px;

  ${c(r.md)} {
    display: none;
  }
`,z=n(f).withConfig({displayName:"ArticlePageDisclaimer"})`
  ${d("typography.definitions.discovery.subhed-section-collection")}
  p {
    ${d("typography.definitions.globalEditorial.context-secondary")};
  }
  ${({theme:e})=>s(e,"color","colors.consumption.body.standard.body")};
  ${c(r.md)} {
    .grid-layout--adrail & {
      grid-column: 1 / -1;
    }
  }
`,q=n("div").withConfig({displayName:"ArticlePageChunksContent"})`
  ${({isNarrowContentWell:e})=>e&&`\n        .grid > *:first-child,\n        .body__container {\n          grid-column: 1 / -1;\n\n          ${c(r.lg)} {\n            grid-column: 4 / span 6;\n          }\n        }\n\n        ${N} {\n          .grid--item {\n            grid-column: 1 / -1;\n          }\n        }\n      `}

  ${({shouldBlurText:e})=>{e&&a`
        .grid:first-of-type .body__container p.has-dropcap::first-letter {
          color: transparent;
        }
        ${C} {
          * {
            text-shadow: 0 0 20px
              rgba(
                ${g("colors.consumption.body.standard.body")},
                0.5
              );
            color: transparent;
          }

          a:not(.button) {
            ${({theme:e})=>u(e,"colors.consumption.body.standard.link","colors.consumption.body.standard.link-hover")};
            text-shadow: 0 0 20px
              rgba(
                ${g("colors.consumption.body.standard.link")},
                0.5
              );
          }
        }
      `}}
`,K=n("article").withConfig({displayName:"ArticlePageMainContent"})`
  ${({shouldBlurText:e})=>e&&"\n        .grid {\n          .body__container {\n            p.has-dropcap,\n            p.has-dropcap__lead-standard-heading {\n              &::first-letter {\n                color: inherit;\n              }\n            }\n          }\n        }\n      "}
`,Z=n(P).withConfig({displayName:"ArticlePageSplitAdRail"})`
  ${$},
  ${$} > ${T} {
    height: 100%;
  }

  > aside > ${$} {
    position: static;
  }
`,X=n("div").withConfig({displayName:"ArticlePageSplitAdRailContent"})`
  display: flex;
  flex-direction: column;
  height: 100%;

  ${$} {
    margin-bottom: 0;
  }

  > div {
    flex: 1;
  }
`,Y=n("div").withConfig({displayName:"ArticlePageSplitAdRailTop"})``,J=n("div").withConfig({displayName:"ArticlePageSplitAdRailMiddle"})`
  margin-top: 40px;
`,Q=n("div").withConfig({displayName:"ArticlePageSplitAdRailBottom"})`
  margin-top: 40px;
`,ee=n("div").withConfig({displayName:"ArticlePageBodyGridContainer"})`
  width: 100%;
`,te=n.div.withConfig({displayName:"ArticlePageChunksGrid"})`
  ${({adRail:e})=>!e&&`\n     > ${I} {\n      grid-column: 1/ -1;\n      ${c(r.md)} {\n        grid-column: 3 / span 8;\n      }\n    }`}
  > ${I} {
    margin-bottom: ${l(2)};
  }

  ${({pageBackgroundTheme:e})=>e&&` .ad.ad--in-content {\n        display: none;\n      }\n      ${O} {\n        margin-top: 2rem;\n      }  \n    `}
`,ie=n.div.withConfig({displayName:"ArticlePageContentFooterGrid"})`
  ${I} {
    grid-column: 1 / -1;
    ${c(r.md)} {
      grid-column: 3 / span 8;
    }
    ${({isNarrow:e})=>e&&`\n         ${c(r.md)} {\n            grid-column: 4 / span 6;\n          }\n         `}
  }
`,ne=n.div.withConfig({displayName:"ArticlePageDisclaimerGrid"})`
  ${({adRail:e})=>!e&&`\n    > ${I} {\n      grid-column: 1/ -1;\n      ${c(r.md)} {\n        grid-column: 3 / span 8;\n      }\n    }`}
`,oe=n.div.withConfig({displayName:"ContentWrapperGrid"})`
  ${({isadRail:e})=>!e&&`\n    > ${I} {\n      grid-column: 1/ -1;\n      ${c(r.md)} {\n        grid-column: 3 / span 8;\n      }\n    }`}
`,ae=n(v).withConfig({displayName:"LinkStackArticlePage"})`
  margin: 48px 0;
  padding-bottom: 0;

  ${w} {
    margin: 0;
  }

  ${S} > ul {
    margin-left: 0;
    padding-left: 0;
  }
`;e.exports={ArticlePageBase:_,ArticlePageGlobalStyle:G,ArticlePageLedeBackground:j,ArticlePageContentBackGround:F,ArticlePageChunks:V,ArticlePageBodyMobileTruncatedBtn:U,ArticlePageChunksContent:q,ArticlePageMainContent:K,ArticlePageDisclaimer:z,ArticlePageSplitAdRail:Z,ArticlePageSplitAdRailContent:X,ArticlePageSplitAdRailTop:Y,ArticlePageSplitAdRailMiddle:J,ArticlePageSplitAdRailBottom:Q,ArticlePageBodyGridContainer:ee,ArticlePageChunksGrid:te,ArticlePageContentFooterGrid:ie,ArticlePageDisclaimerGrid:ne,ContentWrapperGrid:oe,LinkStackArticlePage:ae,PaywallInlineBarrierWithWrapperGrid:W}},1573:function(e,t,i){const{asConfiguredComponent:n}=i(9),o=i(1582);e.exports=n(o,"BreadcrumbTrail")},1574:function(e,t,i){const{default:n,css:o}=i(3),{BREAKPOINTS:a,ZINDEX_MAP:r}=i(6),{applyGridSpacing:l,cssVariablesGrid:s}=i(16),{calculateSpacing:d,minScreen:c,maxScreen:p,minMaxScreen:g,getColorToken:u,getTypographyStyles:m,firstLetterDropCap:h}=i(4),y=i(14),{universalGridCore:b}=i(57),f=i(58),{RowWrapper:C}=i(473),v=i(19),{GridItem:w}=i(25),S=i(489),{SummaryRiverSection:k}=i(471),{SummaryCollageOneGridWithMargin:x}=i(1604),{SectionTitleHed:$}=i(69),T=`\n  ${s()}\n  ${l("padding")}\n\n  margin: 0 auto;\n  width: 100%;\n  max-width: ${a.xxl};\n`,E=n(f).withConfig({displayName:"MultiPackageBaseRow"})`
  &:first-child,
  & ~ & {
    ${({hasMarginTopMultiPackageRow:e})=>e?"":"margin-top: 0;"}
  }

  inline-embed[name='align-right'] {
    text-align: right;
  }

  inline-embed[name='align-center'] {
    text-align: center;
  }
`,N=n(y.Utility).withConfig({displayName:"MultiPackageRow"})`
  &&& {
    align-items: center;
    width: auto;
  }
`,B=n(E).withConfig({displayName:"MultiPackageRow"})`
  ${({hasNoTopMargin:e})=>e?"":`margin-top: ${d(7)};`}

  ${({hasLightBgForLinkBanner:e,hasPlpFilterableContainerLightBackgroundColor:t,theme:i,isInvertedTheme:n})=>e||t?`background-color: ${u(i,"colors.discovery.body.light.background")};`:n?`background-color: ${u(i,"colors.consumption.lead.inverted.background")};`:""}

  ${({hasMediumMargin:e,hasNoBottomMargin:t,hasReducedMargin:i,isNativeAd:n})=>e?`margin-bottom: ${d(4)};`:t||n&&i?"margin-bottom: 0;":i?`margin-bottom: ${d(2)};`:`margin-bottom: ${d(7)};`}

  ${({hasMarginBottomMultiPackageRow:e})=>e?`\n          ${c(a.lg)} {\n            margin-bottom: ${d(5)};\n          }\n          ${g(a.md,a.lg)} {\n            margin-bottom: ${d(4)};\n          }\n          ${g(0,a.md)} {\n            margin-bottom: ${d(3)};\n          }`:""}

  ${({hasStickyLinkBanner:e})=>e?`\n        position: sticky;\n        top: 0px;\n        z-index: ${r.interstitialLayer};\n      `:""} 

  /* TODO: this should be a configuration for a layout
     Specificity is required due to star selector in homepage  */
  ${c(a.lg)} {
    ${C}.homepage__half-column-row + && {
      width: 50%;
      max-width: 800px;
      ${$} {
        ${m("typography.definitions.discovery.subhed-section-secondary")};
      }
    }
    .homepage__half-column-row + && > ${x} {
      padding-left: var(--grid-gap);
    }
    ${k} {
      margin: 0 auto;
      max-width: 1600px;
    }
  }
`,L=n(v.WithMargins).withConfig({displayName:"DiscoveryQuoteRow"})`
  ${({hasTopAndBottomBorderQuote:e,theme:t})=>e?`\n    border-top:1px solid ${u(t,"colors.discovery.body.white.divider")};\n    \n    border-bottom:1px solid ${u(t,"colors.discovery.body.white.divider")};`:""}

  ${w} {
    grid-column: 1/-1;

    ${c(a.md)} {
      grid-column: 3/11;
    }
  }
`,I=n(S).withConfig({displayName:"MultiPackageBody"})`
  p.has-dropcap {
    margin-top: ${d(4,"px")};

    &::first-letter {
      ${h};
    }

    &.has-dropcap__lead-standard-heading {
      &::first-letter {
        padding: 0.1em 0.05em 0 0;
        color: ${u("colors.consumption.lead.standard.heading")};
      }
    }
  }
  ${({constrainPagragraph:e})=>e&&`\n    inline-embed{\n      display: block;\n      p {\n        display:inline-block;\n        max-width: ${d(40.875)};\n        ${c(a.lg)} {\n          max-width: ${d(91.5)};\n        }\n      }\n    }\n  `}
`,A=o`
  &.puzzles-games-landing-page {
    .ticker-wrapper {
      margin-bottom: 0;
    }

    .ticker-view > div:nth-child(2) {
      margin-top: ${d(4)};
    }

    .summary-collage-six-puzzles-games .summary-item:first-child h3 {
      ${m("typography.definitions.discovery.description-core")};
      font-size: ${d(4)};
    }

    .verso-features {
      margin-bottom: ${d(4)};
    }

    > div:nth-child(3)
      .summary-collage-six-puzzles-games
      .summary-list--collection-list {
      ${p(a.lg)} {
        padding-top: 0;
      }
    }

    .summary-river-puzzles-games {
      h2 {
        font-size: 24px;
      }

      & > section {
        .summary-item:last-child {
          border-bottom: 0;
        }

        > div {
          margin-bottom: ${d(2)};

          > div:first-child {
            margin-bottom: 0;
          }
        }
      }

      .summary-item {
        padding-top: ${d(2)};
      }
    }

    .summary-item--is-dense .summary-item__asset-container {
      ${g(0,a.xl)} {
        display: block;
      }

      ${c(a.xl)} {
        float: none;
        margin-left: 0;
      }
    }

    .summary-list--collection-list {
      ${g(0,a.lg)} {
        border-top: 0;
        padding-top: 0;
      }
    }

    .summary-list__items .summary-item:not(:first-child) {
      margin-bottom: 0;
      padding-bottom: ${d(2)};
    }

    .summary-list__items .summary-item:first-child {
      ${g(0,a.lg)} {
        padding-bottom: ${d(2)};
      }

      ${c(a.lg)} {
        padding-bottom: 0;
      }
    }

    .summary-collage-six-puzzles-games .summary-item h3 {
      ${m("typography.definitions.discovery.hed-core-secondary")};
    }

    .summary-collage-six-puzzles-games h2,
    .summary-collection-grid h2 {
      ${m("typography.definitions.discovery.subhed-section-primary")};
    }

    .verso-embed-row inline-embed h1 {
      margin: 0;
      font-size: ${d(5.5)};
    }

    .verso-embed-row {
      margin: ${d(4)} 0;
    }

    ${I} {
      max-width: initial;
    }

    .verso-features h2 {
      font-size: 24px;
    }

    .summary-collage-six-puzzles-games h2 {
      font-size: 20px;
    }

    .summary-item__dek > a {
      ${m("typography.definitions.foundation.link-primary")};
      display: block;
      padding-top: ${d(2.5)};
      text-decoration: none;
      color: ${u("colors.interactive.base.brand-primary")};
    }

    .summary-item__dek > a:hover {
      text-decoration: underline;
    }
  }
`,P=n.div.withConfig({displayName:"MultiPackageContainer"})`
  ${({showFooterAdPadding:e})=>e&&`padding-bottom: ${d(10,"px")};`}

  ${({customClass:e})=>e&&"puzzles-games-landing-page"===e&&A}
  ${({hasMarginBottomMultiPackageRow:e})=>e?"\n            .verso-features {\n              && {\n                margin-bottom: 0;\n              }\n            }\n          ":""}
`,R=n.div.withConfig({displayName:"SectionJumpLinksWrapper"})`
  ${({theme:e})=>(e=>`\n    background: ${u(e,"colors.consumption.body.inverted.display-texture")};\n    ${c(a.md)} {\n      width: ${a.md};\n      padding: ${d(3)} ${d(9)} ${d(5)} ${d(9)};\n    }\n    padding: ${d(2)} ${d(5)} ${d(3.5)} ${d(5)};\n    margin: auto;\n    div {\n      div {\n        h1 {\n          text-align: center;\n        }\n        div {\n          a {\n            font-family: Konnect, helvetica, sans-serif;\n            font-style: normal;\n            line-height: ${d(2.4,"rem")};\n            font-size: ${d(2)};\n            &:not(.button):link,\n            &:not(.button):visited {\n              color: rgb(0, 0, 0);\n            }\n          }\n          display: grid;\n          ${c(a.md)} {\n            grid-template-columns: 1fr 1fr 1fr;\n            grid-row-gap: ${d(1.5)};\n          }\n          grid-template-columns: 1fr 1fr;\n          grid-row-gap: ${d(1)};\n        }\n      }\n    }\n  }\n  `)(e)}
`,D=n(B).withConfig({displayName:"MultiPackageReadMore"})`
  display: flex;
  justify-content: center;
`,M=n.div.withConfig({displayName:"PromoBoxWrapper"})`
  ${T}
`,O=n.div.withConfig({displayName:"EventsListWrapper"})`
  ${s()}

  margin: 0 auto;
  width: 100%;
  max-width: ${a.xxl};

  ${c(a.lg)} {
    padding-right: var(--grid-margin);
    padding-left: var(--grid-margin);
  }
`,H=n.div.withConfig({displayName:"PromoBoxWrapper"})`
  ${T}
`,_=n.div.withConfig({displayName:"NewsletterWrapper"})`
  ${T}

  padding-top: ${d(6)};
  padding-bottom: ${d(6)};

  ${c(a.md)} {
    padding-top: ${d(9)};
    padding-bottom: ${d(9)};
  }
`,G=n.div.withConfig({displayName:"CMUnitWrapper"})`
  ${T}

  ${c(a.lg)} {
    padding-right: var(--grid-margin);
    padding-left: var(--grid-margin);
  }
`,W=n.div.withConfig({displayName:"SubTopicDiscoveryWrapper"})`
  ${b(!0)}
  ${l("padding")}

  margin: 0 auto;
  width: 100%;
  max-width: ${a.xxl};

  ${c(a.lg)} {
    padding-right: var(--grid-margin);
    padding-left: var(--grid-margin);
  }
`,j=n(v).withConfig({displayName:"GridWrapper"})`
  ${T}
`,F=n.div.withConfig({displayName:"EmbedWrapper"})`
  ${T}
`,V=n.div.withConfig({displayName:"TickerWrapper"})`
  ${({isInvertedTheme:e})=>e?`\n        ${s()}\n        ${l("padding",!0)}`:""+T}
`,U=n("div").withConfig({displayName:"MultipackageNoItemsBlock"})`
  ${s()}
  ${l("padding")}
  margin: 0 auto;
  margin-bottom: ${d(4)};
  width: 100%;
  max-width: ${a.xxl};
  color: white;
  font-family: 'LabGrotesque';
  ${({hasRule:e,theme:t})=>e?`\n        &::before {\n          border-top: 1px solid ${u(t,"colors.discovery.body.white.divider")};\n          content: '';\n          grid-column: 1/-1;\n          margin-bottom: ${d(4)};\n          display: block;\n        }\n      `:""}

  h3 {
    margin: 0 auto;
    width: fit-content;
    ${m("typography.definitions.consumptionEditorial.subhed-break-secondary")}
  }

  p {
    font-family: Proxima Nova;
    font-size: 12px;
  }
`,z=n.div.withConfig({displayName:"MultiPackageBodyWrapperGrid"})`
  ${b()}
  ${l("padding")}
`;e.exports={CMUnitWrapper:G,DiscoveryQuoteRow:L,EmbedWrapper:F,GridWrapper:j,EventsListWrapper:O,MultiPackageContainer:P,MultiPackageRow:B,MultiPackageBody:I,MultiPackageReadMore:D,NewsFeedWrapper:H,NewsletterWrapper:_,PromoBoxWrapper:M,SubTopicDiscoveryWrapper:W,TickerWrapper:V,MultipackageNoItemsBlock:U,MultiPackageBodyWrapperGrid:z,SectionJumpLinksWrapper:R,UtilityButton:N}},158:function(e,t,i){const{asVariation:n}=i(12),o=i(159);o.ContentCenterNoBackground=n(o,"ContentCenterNoBackground",{contentAlign:"center",hasBackground:!1}),o.ContentRightNoBackground=n(o,"ContentRightNoBackground",{contentAlign:"right",hasBackground:!1}),o.ContentLeftNoBackground=n(o,"ContentLeftNoBackground",{contentAlign:"left",hasBackground:!1}),e.exports=o},1582:function(e,t,i){const n=i(1),o=i(0),a=i(76),r=i(19),{getSeparatorIconComponent:l}=i(1583),{BreadcrumbTrailWrapper:s,BreadcrumbTrailScrollContainer:d,BreadcrumbTrailItem:c}=i(1586),{trackComponent:p}=i(5),g=({hierarchy:e,shouldRemoveBackgroundColor:t,theme:i,shouldUseContentHeaderColorForLink:n,separatorIcon:g})=>(o.useEffect(()=>{p("BreadcrumbTrail")},[]),o.createElement(a,{palette:i},o.createElement(s,{"data-testid":"BreadcrumbTrail",shouldRemoveBackgroundColor:t},e&&e.length>0&&o.createElement(r.WithMargins,null,o.createElement(d,null,e.map((e,t)=>{const{name:i,slug:a}=e||{};return o.createElement(c,{key:t,shouldUseContentHeaderColorForLink:n},a?o.createElement("a",{className:"breadCrumbLink",href:a},i):o.createElement("span",null,i),l(g))}))))));g.displayName="BreadcrumbTrail",g.defaultProps={separatorIcon:{name:"Chevron",type:"standard"},shouldRemoveBackgroundColor:!1,theme:"standard"},g.propTypes={hierarchy:n.array.isRequired,separatorIcon:n.shape({name:n.string,type:n.oneOf(["standard","thin","thinner"])}),shouldRemoveBackgroundColor:n.bool,shouldUseContentHeaderColorForLink:n.bool,theme:n.string},e.exports=g},1583:function(e,t,i){const n=i(0),{getIconComponent:o}=i(1584);e.exports={getSeparatorIconComponent:e=>{const{name:t,type:i}=e,a=o(t,i)||o("Chevron","standard");return n.createElement(a,Object.assign({},{width:"1rem",height:"1rem"}))}}},1584:function(e,t,i){const n={standard:i(111),thin:i(480),thinner:i(1585)};e.exports={getIconComponent:(e,t="standard")=>{if(!e)return null;const i=n[t][e];return i||null}}},1585:function(e,t,i){e.exports={Bookmark:i(167),BookmarkActivated:i(104),Email:i(474),Facebook:i(475),Twitter:i(477),Print:i(478),Shopping:i(476)}},1586:function(e,t,i){const{default:n}=i(3),{BaseText:o}=i(10),{calculateSpacing:a,getLinkStyles:r,getColorStyles:l}=i(4),{BREAKPOINTS:s}=i(6),{isInverted:d}=i(31),c=n.div.withConfig({displayName:"BreadcrumbTrailWrapper"})`
  ${({theme:e,shouldRemoveBackgroundColor:t})=>{const i=d(e)?"colors.background.black":"colors.background.light";return t?"background-color: transparent;":l(e,"background-color",i)+";"}};

  padding-top: ${a(2.4)};
  padding-bottom: ${a(2.4)};
  width: 100%;

  @media (max-width: ${s.md}) {
    overflow-y: hidden;
    overflow-x: scroll;

    &::-webkit-scrollbar {
      display: none;
    }
  }
`,p=n.div.withConfig({displayName:"BreadcrumbTrailScrollContainer"})`
  display: flex;
  width: max-content;
`,g=n(o).withConfig({displayName:"BreadcrumbTrailItem"})`
  display: inline-flex;
  flex-direction: row;
  align-items: center;

  ${({theme:e})=>d(e)&&`\n      ${l(e,"color","colors.consumption.lead.inverted.link")};`}

  a:active,
  a:link {
    text-decoration: none;
  }

  a:hover,
  a:focus {
    text-decoration: underline;
  }

  .icon {
    margin: 0 ${a(.2)};

    path {
      ${({theme:e})=>d(e)&&`\n          ${l(e,"fill","colors.consumption.lead.inverted.link")};\n        `}
    }
  }

  &:last-of-type {
    a {
      ${({theme:e,shouldUseContentHeaderColorForLink:t})=>{const i=d(e)?r(e,"colors.consumption.lead.inverted.link",null):r(e,"colors.discovery.lead.secondary.link",null);return t?r(e,"colors.consumption.lead.standard.context-signature",null):i}}

      &:active,
      &:link {
        text-decoration: none;
      }

      &:hover,
      &:focus {
        text-decoration: underline;
      }
    }

    span {
      ${({theme:e})=>l(e,"color","colors.discovery.lead.secondary.link")};
    }

    .icon {
      display: none;
    }
  }
`;g.defaultProps={typeIdentity:"typography.definitions.globalEditorial.tags"},e.exports={BreadcrumbTrailWrapper:c,BreadcrumbTrailScrollContainer:p,BreadcrumbTrailItem:g}},1587:function(e,t,i){Object.defineProperty(t,"__esModule",{value:!0});const n=i(2);t.default=n.defineMessages({bylineBrandXAdvertiser:{id:"SponsoredContentHeader.BylineBrandXAdvertiser",defaultMessage:"{brandName} X",description:"Byline text when it's a brand and an advertiser"},bylineBrandedContent:{id:"SponsoredContentHeader.BylineBrandedContent",defaultMessage:"Branded Content By",description:"Byline text for branded content"},bylineCreated:{id:"SponsoredContentHeader.BylineCreated",defaultMessage:"Created By {brandName} For",description:"Byline text for created by brand"},bylinePaidPost:{id:"SponsoredContentHeader.BylinePaidPost",defaultMessage:"PAID POST",description:"Byline text for a paid post"},bylineProduced:{id:"SponsoredContentHeader.BylineProduced",defaultMessage:"Produced By",description:"Byline text for produced by"},bylineProducedByAdvertiser:{id:"SponsoredContentHeader.BylineProducedByAdvertiser",defaultMessage:"Produced By",description:"Byline text for produced by advertiser"},bylineProducedByBrand:{id:"SponsoredContentHeader.BylineProducedByBrand",defaultMessage:"Produced By {brandName} With",description:"Byline text for produced by brand"},bylineSponsored:{id:"SponsoredContentHeader.BylineSponsored",defaultMessage:"Sponsored content",description:"Byline text for sponsored content"},bylineSponsoredContent:{id:"SponsoredContentHeader.BylineSponsoredContent",defaultMessage:"Sponsored Content By",description:"Byline text for sponsored content with a sponsor name"},bylineInCollaboration:{id:"SponsoredContentHeader.BylineInCollaboration",defaultMessage:"In Collaboration With",description:"Byline text for in collaboration with"},bylineSponsoredBy:{id:"SponsoredContentHeader.BylineSponsoredBy",defaultMessage:"Sponsored By",description:"Byline text for sponsored by"},bylineInPartnership:{id:"SponsoredContentHeader.BylineInPartnership",defaultMessage:"In Partnership With",description:"Byline text for in partnership with"},bylineAdvertisementFeatureWith:{id:"SponsoredContentHeader.BylineAdvertisementFeatureWith",defaultMessage:"Advertisement Feature With",description:"Byline text for advertisement feature with"},bylineOriginalContentBy:{id:"SponsoredContentHeader.BylineOriginalContentBy",defaultMessage:"Original Content By",description:"Byline text for Original Content By"},sponsoredLinkCTA:{id:"SponsoredContentHeader.SponsoredLinkCTA",defaultMessage:"Click to go to {sponsorName}'s website",description:"Call to action for sponsored link"}})},159:function(e,t,i){const n=i(1),o=i(8),a=i(0),r=i(11),{trackComponent:l}=i(5),{UtilityLedeHeader:s,UtilityLedeWrapper:d,UtilityLedeHedText:c,UtilityLedeDekText:p,UtilityLedeImage:g}=i(129),u=({ariaLive:e,className:t,dangerousDek:i,dangerousHed:n,image:r,shouldUseAlternativeStyle:u,hasInverted:m,variations:h,variationName:y})=>{a.useEffect(()=>{l("UtilityLede",y)},[y]);const b=r&&Object.keys(r).length>0;return a.createElement(s,{className:o("utility-lede",t),"aria-live":e,"aria-label":"UtilityPageHeader",contentAlign:h.contentAlign,hasBackground:h.hasBackground,hasImage:b,alternativeStyle:u},r&&a.createElement(g,Object.assign({hasImage:b},r)),a.createElement(d,{alternativeStyle:u},a.createElement(c,{"data-testid":"UtilityLedeHedText",hasImage:b,dangerouslySetInnerHTML:{__html:n},hasInverted:m}),i&&a.createElement(p,{"data-testid":"UtilityLedeDekText",hasImage:b,dangerouslySetInnerHTML:{__html:i}})))};u.propTypes={ariaLive:n.string,className:n.string,dangerousDek:n.string,dangerousHed:n.string.isRequired,hasInverted:n.bool,image:n.shape(r.propTypes),shouldUseAlternativeStyle:n.bool,variationName:n.string,variations:n.shape({contentAlign:n.oneOf(["center","left","right"]),hasBackground:n.bool})},u.defaultProps={shouldUseAlternativeStyle:!1,variations:{contentAlign:"center",hasBackground:!0}},e.exports=u},1604:function(e,t,i){const{default:n,css:o}=i(3),{BREAKPOINTS:a}=i(6),{applyGridSpacing:r,cssVariablesGrid:l}=i(16),{RubricLink:s}=i(73),{calculateSpacing:d,minScreen:c,minMaxScreen:p,getTypographyStyles:g,getColorToken:u}=i(4),{BaseLink:m}=i(10),{SocialIconsList:h}=i(28),{SpanWrapper:y}=i(85),b=n.div.withConfig({displayName:"SummaryCollageOneTitle"})`
  grid-column: 1 / -1;

  ${({isSingleFeature:e})=>!e&&`\n      margin-bottom: ${d(2)};\n\n      ${c(a.md)} {\n        margin-bottom: ${d(1)};\n      }\n\n      ${c(a.lg)} {\n        margin-bottom: ${d(0)};\n      }\n    `}
`,f=n.div.withConfig({displayName:"SummaryCollageOneCtaLink"})`
  grid-column: 1 / -1;
  text-align: center;

  span {
    display: block;
    width: 100%;
    text-align: center;
  }
`,C=n(m).withConfig({displayName:"SummaryCollageOneAnchorLink"})`
  ${g("typography.definitions.foundation.link-primary")}
  display: inline-block;
  margin-bottom: ${d(2)};
  vertical-align: top;
  ${c(a.lg)} {
    margin-bottom: ${d(4)};
  }
`;C.defaultProps={colorSecondaryLinkToken:"colors.interactive.base.dark",colorStaticLinkToken:"colors.interactive.base.brand-primary",linkStyle:"global"};const v=o`
  ${l()}
  ${r("padding")}

  display: grid;
  grid-template-columns: repeat(4, 1fr);
  column-gap: var(--grid-gap);
  margin: 0 auto;
  width: 100%;
  max-width: ${a.xxl};
  row-gap: var(--grid-gap);

  .summary-item--layout-placement-text-below {
    &.summary-item--text-align-left,
    &.summary-item--text-align-center {
      .summary-item__hed {
        ${g("typography.definitions.discovery.hed-break-out")};
      }

      .summary-item__dek {
        ${g("typography.definitions.discovery.description-feature")};
      }
    }
  }

  .summary-item__hed-link {
    &::after {
      border-bottom: 1px solid
        ${u("colors.discovery.body.white.accent")};
    }
  }

  .summary-item__content:empty {
    display: none;
  }

  ${h} {
    justify-content: center;
  }

  ${c(a.md)} {
    grid-template-columns: repeat(12, 1fr);
  }
`,w=n.div.withConfig({displayName:"SummaryCollageOneItemComponent"})`
  grid-column: 1 / -1;

  &&& {
    border-bottom: 0;

    .summary-item__content {
      margin: 0;
      @media (min-width: 0) and (max-width: ${a.lg}) {
        margin: 0;
      }
    }
    @media (min-width: 0) and (max-width: ${a.md}) {
      padding-bottom: 0;
    }
  }
`,S=o`
  &&& {
    grid-gap: ${d(4)};
    grid-template-rows: auto 1fr auto;
    height: 100%;
  }

  ${w} {
    display: grid;
    align-items: center;
  }

  ${y}.summary-item__image {
    .summary-item__image {
      display: grid;
      justify-items: center;

      picture {
        max-width: 300px;
        ${c(a.lg)} {
          max-width: 400px;
        }
      }
    }

    .responsive-cartoon__caption {
      margin-top: 0;
      ${c(a.lg)} {
        margin-top: ${d(2)};
        width: 85%;
      }
    }
  }
`,k=o`
  ${p(0,a.lg)} {
    .summary-item__asset-container {
      ${r("margin",!0)};
    }
  }
`,x=o`
  &&& {
    grid-template-rows: unset;
    background-color: ${u("colors.discovery.body.dark.background")};
    max-width: ${a.xxl};

    ${p(a.sm,a.md)} {
      column-gap: ${d(9)};
      margin: ${d(0)};
      padding: ${d(0)};
    }
    ${c(a.md)} {
      column-gap: ${d(3)};
      margin: ${d(0)};
      padding: ${d(5)} ${d(3)};
    }
    ${c(a.lg)} {
      margin: ${d(0)};
      padding: ${d(7)} ${d(6)};
    }
    ${c(a.xl)} {
      column-gap: 2rem;
      margin: ${d(0)};
      padding: ${d(10)} ${d(8)};
    }
    ${c(a.xxl)} {
      column-gap: 2rem;
      margin: auto;
      padding: ${d(10)} ${d(8)};
    }
  }
  ${s} {
    color: ${u("colors.discovery.body.dark.context-signature")};
  }

  .summary-item__rubric {
    color: ${u("colors.discovery.body.dark.context-signature")};
  }

  .summary-item__dek {
    color: ${u("colors.discovery.body.dark.description")};
  }

  .summary-item__hed--hed-core-primary {
    ${g("typography.definitions.discovery.hed-core-primary")}
    color: ${u("colors.discovery.body.dark.heading")};
  }

  .summary-item__hed {
    color: ${u("colors.discovery.body.dark.heading")};
  }

  .summary-item--dark-background-right {
    ${c(a.md)} {
      grid-column: 2 / -2;
      margin: ${d(0)} ${d(-3)};
    }
    ${c(a.lg)} {
      grid-column: 2 / -2;
      margin: ${d(0)} ${d(-3)};
      padding: ${d(0)};
    }
    ${c(a.xl)} {
      grid-column: 2 / -2;
      margin: ${d(0)} ${d(-4)};
      padding: ${d(0)};
    }
    ${c(a.xxl)} {
      grid-column: 2 / -2;
      margin: ${d(0)};
      padding: ${d(0)};
    }

    .summary-item__asset-container {
      ${c(a.md)} {
        margin-left: ${d(-3)};
      }
      ${c(a.xl)} {
        margin-left: ${d(0)};
      }
    }
  }

  .summary-item__content {
    margin: ${d(0)};
    ${c(a.md)} {
      padding-right: ${d(3)};
    }
    ${c(a.lg)} {
      padding-right: ${d(5)};
    }
    ${c(a.xl)} {
      padding-right: ${d(3)};
    }
    ${c(a.xxl)} {
      padding-right: ${d(6)};
    }
  }
`,$=n.div.withConfig({displayName:"SummaryCollageOneGridWithMargin"})`
  ${v}
  ${({isSingleFeature:e})=>e&&S}

  ${({hasFullWidthImage:e})=>e&&k}
  
  ${({isFullBleedDarkBackground:e})=>e&&x}
`,T=n.div.withConfig({displayName:"SummaryCollageOneIsFullBleed"})`
  ${({isFullBleedDarkBackground:e})=>e&&o`
      background-color: ${u("colors.discovery.body.dark.background")};
      max-width: 100%;
    `}
`;e.exports={SummaryCollageOneTitle:b,SummaryCollageOneCtaLink:f,SummaryCollageOneGridWithMargin:$,SummaryCollageOneIsFullBleed:T,SummaryCollageOneItemComponent:w,SummaryCollageOneAnchorLink:C}},1617:function(e,t,i){const n=i(0),o=i(1),{trackComponent:a}=i(5),r=({children:e,name:t})=>{if(n.useEffect(()=>{a("Slot")},[]),!t)throw new Error("A slot must contain a name!");return e};r.propTypes={children:o.node.isRequired,name:o.string.isRequired};e.exports={Slot:r,getSlots:(e,t=[])=>{const i=new Set(t),o={};let a;const l=[];return n.Children.forEach(e,e=>{a=e.props.name,l.push(e.props.url),e.type===r&&a&&(0===i.size||i.has(a))&&(o[a]=e)}),{slots:o,urlData:l}}}},1618:function(e,t,i){const{asConfiguredComponent:n}=i(9),o=i(1619);e.exports=n(o,"FilterableSummaryList")},1619:function(e,t,i){const n=i(0),o=i(35),a=i(1),{useIntl:r}=i(2),{calculateSpacing:l}=i(4),{getSlots:s}=i(1519),d=i(241),c=i(248),p=i(40),{Disclaimer:g}=i(97),u=i(100),{trackComponent:m}=i(5),{componentTracking:h,googleAnalytics:y}=i(13),{CarouselWrapper:b,ChipWrapper:f,EditorCard:C,EditorDescription:v,EditorDetails:w,EditorName:S,EditorResponsiveAsset:k,EditorsPicksCarousel:x,EditorTitle:$,FilterableSummaryListDynamicGridItemLayout:T,FilterableSummaryListGrid:E,FilterableSummaryListWrapper:N,TitleWrapper:B,TitleToggleChipListWrapper:L}=i(1520),{ArticleGalleryCarouselBtnWrapper:I,ArticleGalleryCarouselButton:A}=i(122),P=i(1620).default,R={spacing:{sm:l(1.5),xl:l(2)},width:{sm:`calc(60% - ${l(1.5)})`,lg:`calc(27% - ${l(1.5)})`,xl:`calc(24% - ${l(2)})`,xxl:`calc(21% - ${l(2)})`}},D=({affiliateDisclaimer:e,buttonConfig:t,children:i,className:a,controlButtonIcon:l,controlPlacement:D,controlPosition:M,dangerousDek:O,dangerousHed:H,defaultToggleChipTitle:_,editor:G,hasBorderBottom:W,hasGridBottomPadding:j,hasCarouselSliderPagination:F,hasControls:V,hasCustomSlider:U,hasImpressionTracking:z,hasNavigationButtonVariation:q,hasNoHorizontalScrollCarousel:K,paginationStyle:Z,hasPadding:X,hasPagination:Y,hasProductDisclaimerAlternativeStyle:J,hasTitleMarginTop:Q,hasTitleNoMargin:ee,hasToggleGridColor:te,hasHigherHorizontalPadding:ie,sectionTitleVariation:ne,selectedToggleChipTitle:oe,shouldAppendFilterInUrl:ae,shouldDisplaySingleSlot:re,shouldNotDisplayAllLabel:le,shouldEnableBundleComponentAnalytics:se,shouldEnableProductDisclaimer:de,shouldUseAlternativeTitleStyle:ce,hasCarouselControl:pe,toggleChipRole:ge,trackingNamespace:ue,isDotClickable:me,layout:he,pos:ye,gridConfig:be,isEditorsPicksCarousel:fe})=>{n.useEffect(()=>{m("FilterableSummaryList")},[]);const{formatMessage:Ce}=r(),{slots:ve,urlData:we}=s(i),Se=Object.keys(ve),[ke,xe]=n.useState(Se),[$e,Te]=n.useState(ke[0]),[Ee,Ne]=n.useState(_),Be=H||O,Le=(null==ue?void 0:ue.toggle)||H,Ie=(null==ue?void 0:ue.card)||H;if(0===ke.length)return null;const Ae=ke.length>1||re,Pe=t.hasCtaLink?Ce(P.atArticleGalleryCarouselBtnTextWithCtaLink,{categoryName:t.name}):Ce(P.atArticleGalleryCarouselBtnText,{categoryName:t.name}),{isDynamicGridItemLayout:Re,shouldDisplayDenseGrid:De}=be||{},Me=()=>n.Children.map(ve[$e].props.children,(e,t)=>{const i=h.addDataSectionTitleAttribute(se,`${Ie}/${$e}/`,t,!1),o=n.cloneElement(e,{analyticsDataAttribute:i});return n.createElement("div",null,o)}),Oe=()=>n.createElement(b,null,n.createElement(g.TextCenterNoTopRule,{isEnabled:de&&pe,hasHigherHorizontalPadding:ie,hasProductDisclaimerAlternativeStyle:J,disclaimerHtml:e}),n.createElement(u,{hasControls:V,hasNavigationButtonVariation:q,hasPagination:Y,controlButtonIcon:q?"ArrowIcon":l,controlPlacement:D,controlPosition:M,isDotClickable:me,hasPadding:X,hasNoHorizontalScrollCarousel:K,hasImpressionTracking:z,paginationStyle:F&&"slider"===Z?Z:"bullet",hasCustomSlider:F&&U,dangerousHed:Be,pos:ye},n.Children.map(ve[$e].props.children,(e,t)=>{const i=h.addDataSectionTitleAttribute(se,`${Ie}/${$e}/`,t,!1),o=n.cloneElement(e,{analyticsDataAttribute:i}),a=`${Be}/${$e}`;return n.createElement(u.CarouselItem,Object.assign({},R,{key:`${$e}-${t}`,carouselTitle:a,carouselItemIndex:t,carouselItemName:e.props.dangerousHed,pos:ye}),o)})),t.showButton&&t.url&&n.createElement(I,null,n.createElement("div",{className:"more-products"},n.createElement(A,{className:"article-gallery__more-button",label:Pe,btnStyle:"outlined",ariaLabel:Pe,href:t.hasCtaLink?t.url:"/products/shop"+t.url,inputKind:"link"})))),He=(e,t)=>{var i,n;Te(t),re&&(e.detail.checked?(xe([t]),oe&&Ne(oe)):(Te(Se[0]),xe(Se),Ne(_))),i=t,n="body",y.emitUniqueGoogleTrackingEvent("toggle-click",{clickText:i,clickType:n})};return n.useEffect(()=>{var e;{const t=decodeURIComponent(null===(e=null===window||void 0===window?void 0:window.location)||void 0===e?void 0:e.hash),i=ke.findIndex(e=>"#"+e.toLowerCase()===t.toLowerCase());Te(ke[i>=0?i:0])}},[ke]),n.createElement(N,{className:a,hasToggleGridColor:te,hasPadding:X,hasBorderBottom:W},n.createElement(L,{shouldUseAlternativeTitleStyle:ce},Be&&n.createElement(B,{className:"section-title",hasPadding:X,hasTitleMarginTop:Q,as:p[ne],dangerousHed:H,dangerousDek:O,hasTitleNoMargin:ee}),Ae&&n.createElement(f,{className:"clip-list",hasToggleGridColor:te,hasPadding:X},n.createElement(c,{contentAlign:"center",layout:"nowrap",hasToggleGridColor:te,label:Ee},ke.map((e,t)=>{const i=h.addDataSectionTitleAttribute(se,`${Le}/${e}`),o=e===$e;return le&&"All"===e?null:n.createElement(d,{analyticsDataAttribute:i,key:e,isChecked:o,hasToggleGridColor:te,onChange:t=>He(t,e),isAnchorUrl:ae,redirectUrl:we[t],shouldDisplaySingleChip:re,role:ge},e)})))),fe&&!o(G)?n.createElement(x,null,n.createElement(C,null,G.editorPhoto&&n.createElement(k,Object.assign({},G.editorPhoto)),n.createElement(w,null,n.createElement(S,null,G.name),n.createElement($,null,G.title)),n.createElement(v,null,G.editorNote)),Oe()):function(){switch(he){case"GridFourColumns":return Re?n.createElement(T,{shouldDisplayDenseGrid:De,hasGridBottomPadding:j},Me()):n.createElement(E,null,Me());case"FullBleed":return n.createElement("div",null,Me());default:return Oe()}}())};D.propTypes={affiliateDisclaimer:a.string,buttonConfig:a.object,children:a.node.isRequired,className:a.string,controlButtonIcon:a.oneOf(["ChevronIcon","ArrowIcon"]),controlPlacement:a.oneOf(["right","space-between"]),controlPosition:a.oneOf(["top","bottom","center"]),dangerousDek:a.string,dangerousHed:a.string,defaultToggleChipTitle:a.string,editor:a.object,gridConfig:a.object,hasBorderBottom:a.bool,hasCarouselControl:a.bool,hasCarouselSliderPagination:a.bool,hasControls:a.bool,hasCustomSlider:a.bool,hasGridBottomPadding:a.bool,hasHigherHorizontalPadding:a.bool,hasImpressionTracking:a.bool,hasNavigationButtonVariation:a.bool,hasNoHorizontalScrollCarousel:a.bool,hasPadding:a.bool,hasPagination:a.bool,hasProductDisclaimerAlternativeStyle:a.bool,hasTitleMarginTop:a.bool,hasTitleNoMargin:a.bool,hasToggleGridColor:a.bool,isDotClickable:a.bool,isEditorsPicksCarousel:a.bool,layout:a.string,paginationStyle:a.string,pos:a.number,sectionTitleVariation:a.string,selectedToggleChipTitle:a.string,shouldAppendFilterInUrl:a.bool,shouldDisplaySingleSlot:a.bool,shouldEnableBundleComponentAnalytics:a.bool,shouldEnableProductDisclaimer:a.bool,shouldNotDisplayAllLabel:a.bool,shouldUseAlternativeTitleStyle:a.bool,toggleChipRole:a.string,trackingNamespace:a.shape({toggle:a.string,card:a.string})},D.displayName="FilterableSummaryList",D.defaultProps={buttonConfig:{name:"",showButton:!1,url:""},controlButtonIcon:"ChevronIcon",controlPlacement:"space-between",controlPosition:"center",hasControls:!0,hasHigherHorizontalPadding:!1,hasImpressionTracking:!1,hasNoHorizontalScrollCarousel:!1,hasPagination:!0,hasTitleMarginTop:!1,isDotClickable:!1,isEditorsPicksCarousel:!1,shouldAppendFilterInUrl:!0,shouldDisplaySingleSlot:!1,shouldEnableBundleComponentAnalytics:!1,shouldEnableProductDisclaimer:!1,shouldNotDisplayAllLabel:!1,shouldUseAlternativeTitleStyle:!1},D.displayName="FilterableSummaryList",e.exports=D},1620:function(e,t,i){Object.defineProperty(t,"__esModule",{value:!0});const n=i(2);t.default=n.defineMessages({atArticleGalleryCarouselBtnText:{id:"FilterableSummaryList.AtArticleGalleryCarouselBtnText",defaultMessage:"VIEW ALL {categoryName}",description:"Article and Gallery carousel button text"},atArticleGalleryCarouselBtnTextWithCtaLink:{id:"FilterableSummaryList.AtArticleGalleryCarouselBtnTextWithCtaLink",defaultMessage:"{categoryName}",description:"Article and Gallery carousel button text for cta link"}})},164:function(e,t,i){Object.defineProperty(t,"__esModule",{value:!0});const n=i(2);t.default=n.defineMessages({showAllPhotos:{id:"ContentHeader.ShowAllPhotos",defaultMessage:"Show all Photos",description:"Call to action to view entire photo gallery"},readReviews:{id:"ContentHeader.ReadReviews",defaultMessage:"Read Reviews",description:"Call to action to read reviews"}})},1662:function(e,t,i){const{asConfiguredComponent:n}=i(9);e.exports=n(i(1663),"SponsoredContentHeader")},1663:function(e,t,i){const n=i(8),o=i(1),a=i(0),{useIntl:r}=i(2),l=i(1587).default,{trackComponent:s}=i(5),{SponsoredContentHeaderWrapper:d,SponsoredContentHeaderExternalLink:c,SponsoredContentHeaderInfoBox:p,SponsoredContentHeaderBylineText:g,SponsoredContentHeaderResponsiveAsset:u,SponsoredContentHeaderSponsorName:m}=i(1664),{getBylineText:h,getLogoRatio:y,getSponsoredHeaderDisplayOptions:b,getValidBylineOption:f}=i(1665),C=({brandName:e,bylineOption:t,bylineVariant:i,campaignUrl:o,className:C,sponsorLogo:v,sponsorName:w})=>{a.useEffect(()=>{s("SponsoredContentHeader")},[]);const S=r(),k=f(t),{isBrandedLegacy:x,shouldDisplayLogo:$}=b({bylineOption:k,bylineVariant:i,hasLogo:!!v}),T=h({intl:S,bylineOption:k,brandName:e}),E=y({sponsorLogo:v});return a.createElement(d,{isBrandedLegacy:x,className:n(C,k.replace("_","-")),"data-testid":"SponsoredContentHeaderWrapper"},a.createElement(c,{additionalRelVals:["sponsored"],href:o||void 0,attributes:{"aria-label":S.formatMessage(l.sponsoredLinkCTA,{sponsorName:w})}},a.createElement(p,{isBrandedLegacy:x},a.createElement(g,{isBrandedLegacy:x,"data-testid":"SponsoredContentHeaderBylineText"},T),$?a.createElement(u,{altText:v.altText,constrainLogoByWidth:E>1,isBrandedLegacy:x,sources:v.sources}):a.createElement(m,{isBrandedLegacy:x},w))))};C.propTypes={brandName:o.string.isRequired,bylineOption:o.string.isRequired,bylineVariant:o.string.isRequired,campaignUrl:o.string.isRequired,className:o.string,sponsorLogo:o.any,sponsorName:o.string.isRequired},C.displayName="SponsoredContentHeader",e.exports=C},1664:function(e,t,i){const n=i(3).default,{BaseText:o}=i(10),{calculateSpacing:a,getColorStyles:r,getTypographyStyles:l}=i(4),s=i(78),d=i(34),c=n.div.withConfig({displayName:"SponsoredContentHeaderWrapper"})`
  display: flex;
  justify-content: center;
  ${({theme:e})=>r(e,"background-color","colors.discovery.body.light.background")};
  padding: ${a(2)};
  width: 100%;
  min-height: 80px;

  ${({isBrandedLegacy:e})=>e?`\n    grid-column: 1 / -1;\n    padding: unset;\n    height: 60px;\n    min-height: unset;\n\n    &.light-theme {\n      ${({theme:e})=>r(e,"background-color","colors.background.light")}\n    }\n  `:""}
`,p=n(d).withConfig({displayName:"SponsoredContentHeaderExternalLink"})`
  text-decoration: none;
`,g=n.div.withConfig({displayName:"SponsoredContentHeaderInfoBox"})`
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100%;

  ${({isBrandedLegacy:e})=>e?"& { flex-direction: unset; }":""}
`,u=n(o).withConfig({displayName:"SponsoredContentHeaderBylineText"})`
  ${({theme:e,isBrandedLegacy:t})=>t?`\n      ${l(e,"typography.definitions.globalEditorial.context-primary")};\n      display: flex;\n      flex-direction: row;\n      align-items: center;\n      justify-content: flex-end;\n      padding-right: ${a(2)};\n      height: 100%;\n\n      &.light-theme {\n        ${({theme:e})=>r(e,"color","colors.discovery.body.light.heading")}\n      }\n    }\n  `:""}
`;u.defaultProps={as:"div",colorToken:"colors.consumption.lead.standard.syndication",typeIdentity:"typography.definitions.globalEditorial.syndication"};const m=n(s).withConfig({displayName:"SponsoredContentHeaderResponsiveAsset"})`
  &.responsive-asset {
    display: flex;
    align-items: center;
    margin-top: ${a(1)};
    overflow: visible;

    ${({theme:e,isBrandedLegacy:t})=>t?`\n      justify-content: flex-start;\n      margin-top: unset;\n      padding-left: ${a(2)};\n      border-left: 1px solid;\n      ${r(e,"border-color","colors.discovery.body.light.divider")};\n    `:""}
  }

  &.responsive-image {
    height: 60px;

    img {
      height: 100%;
    }

    ${({constrainLogoByWidth:e})=>e?"{\n      width: 60px;\n      height: unset;\n\n      img {\n        height: unset;\n      }\n    }":""}
  }
`,h=n(o).withConfig({displayName:"SponsoredContentHeaderSponsorName"})`
  display: flex;
  align-items: center;
  margin-top: ${a(.5)};

  ${({isBrandedLegacy:e,theme:t})=>e?`\n    justify-content: flex-start;\n    margin-top: unset;\n    padding-left: ${a(.5)};\n\n    &.light-theme {\n      ${r(t,"color","colors.discovery.body.light.syndication")};\n    }\n  `:""}
`;h.defaultProps={as:"div",colorToken:"colors.consumption.lead.standard.syndication",typeIdentity:"typography.definitions.consumptionEditorial.description-feature"},e.exports={SponsoredContentHeaderWrapper:c,SponsoredContentHeaderExternalLink:p,SponsoredContentHeaderInfoBox:g,SponsoredContentHeaderBylineText:u,SponsoredContentHeaderResponsiveAsset:m,SponsoredContentHeaderSponsorName:h}},1665:function(e,t,i){const n=i(1587).default,o={brand_x_advertiser:n.bylineBrandXAdvertiser,branded_content:n.bylineBrandedContent,created:n.bylineCreated,original_content_by:n.bylineOriginalContentBy,paid_post:n.bylinePaidPost,produced:n.bylineProduced,produced_by_advertiser:n.bylineProducedByAdvertiser,produced_by_brand:n.bylineProducedByBrand,sponsored:n.bylineSponsored,sponsored_content:n.bylineSponsoredContent,in_collaboration:n.bylineInCollaboration,sponsored_by:n.bylineSponsoredBy,in_partnership:n.bylineInPartnership,advertisement_feature_with:n.bylineAdvertisementFeatureWith};function a(e){return Object.prototype.hasOwnProperty.call(o,e)?e:"produced_by_advertiser"}e.exports={BYLINE_TEMPLATES:o,getBylineText:function({intl:e,bylineOption:t="produced_by_advertiser",brandName:i}){return e.formatMessage(o[t],{brandName:i})},getLogoRatio:function({sponsorLogo:e}){var t,i,n,o;return((null===(i=null===(t=null==e?void 0:e.sources)||void 0===t?void 0:t.sm)||void 0===i?void 0:i.height)||0)/((null===(o=null===(n=null==e?void 0:e.sources)||void 0===n?void 0:n.sm)||void 0===o?void 0:o.width)||1)},getSponsoredHeaderDisplayOptions:function({bylineOption:e,bylineVariant:t,hasLogo:i}){const n=a(e),o="sponsored"===n||"produced"===n;return{isBrandedLegacy:o,shouldDisplayLogo:i&&("logo"===t||o)}},getValidBylineOption:a}},1666:function(e,t,i){const{asConfiguredComponent:n}=i(9),o=i(1667);e.exports=n(o,"VersoFilterableSummaryList")},1667:function(e,t,i){const n=i(0),o=i(1),a=i(133),r=i(45),{useIntl:l}=i(2),{Slot:s}=i(1519),d=i(481),c=i(1668),p=i(1671),{FilterableSummaryList:g}=i(1672),u=i(262),m=i(42),{trackComponent:h}=i(5),{modifyItemObjectBasedOnContentType:y,childTobeRendered:b}=i(1673),{formatGtmData:f,productImpressionTracking:C}=i(121),v=({copilotId:e,buttonConfig:t,ctaCardAspectRatio:i,dangerousHed:o,dangerousDek:v,dropShipSellers:w,editor:S,groups:k,hasBorder:x,hasBorderBottom:$,hasCarouselSliderPagination:T,hasControls:E,hasImageGrid:N,hasAtRetailerNameLowerCase:B,hasNavigationButtonVariation:L,hasPadding:I,hasProductPriceColor:A,hasProductNewPriceColor:P,hasProductWhiteBackground:R,hasGridColumn:D,hasUnderlineHed:M,isProductCardName:O,isProductCardRetailerName:H,shouldCheckProductInView:_,hasMarginTopAuto:G,hasPLPBrandNameContextTitle:W,hasPLPCardNameDescriptionCore:j,paginationStyle:F,sectionTitleVariation:V,shouldEnableBundleComponentAnalytics:U,shouldHideDangerousDek:z,shouldHidePublishDate:q,shouldPlayInline:K,shouldRenderCtaCard:Z,showOfferUrl:X,summaryItemVariation:Y,shouldUseAlternativeTitleStyle:J,shouldUseProductPriceSecondary:Q,expVariationName:ee,isDropshipProduct:te,trackingNamespace:ie,isDotClickable:ne,layout:oe,gridConfig:ae,priceFormatting:re})=>{n.useEffect(()=>{h("VersoFilterableSummaryList")},[]);const{formatMessage:le}=l(),se=oe||"Carousel",de="ProductCarousel"===se,ce="GridFourColumns"===se?"PLP":oe,pe=o?o.replace(/[^a-zA-Z]+/gi,"-").toLowerCase():"",{gridItemColSpan:ge}=ae||{};n.useEffect(()=>{de||(window.addEventListener("scroll",r(()=>{C(document.getElementsByClassName("impressionTracking"),ce)},1e3)),window.addEventListener("load",()=>C(document.getElementsByClassName("impressionTracking"),ce)))},[ce,de]);function ue(t){switch(se){case"ProductCarousel":case"GridFourColumns":case"EditorsPicksCarousel":return(t=>t.items.map((t,i)=>{const o=t.contentType;if("product"===o)return n.createElement(u,Object.assign({},t,{key:`${t.dangerousHed}-${i}`,hasAtRetailerNameLowerCase:B,hasImageGrid:N,hasProductPriceColor:A,hasProductNewPriceColor:P,shouldUseProductPriceSecondary:Q,hasProductWhiteBackground:R,isLazy:!0,isProductCardName:O,isProductCardRetailerName:H,hasMarginTopAuto:G,hasPLPBrandNameContextTitle:W,hasPLPCardNameDescriptionCore:j,hasUnderlineHed:M,hasCarouselControl:de,shouldCheckProductInView:_,layout:se,copilotID:e,dropShipSellers:w,isDropshipProduct:te,showOfferUrl:X,hasImpressionTracking:!0,data_item:t,onClick:()=>f(window,Object.assign(Object.assign({},t),{expVariationName:ee}),i,ce,pe),expVariationName:ee,layoutName:pe,priceFormatting:re}));const a=y(t,o,K),r=m.TextBelowLeft;return n.createElement(r,Object.assign({gridItemColSpan:ge,shouldHideIcon:!0,shouldHideMetadataSecondary:!0},a,{key:`${t.dangerousHed}-${i}`,hasBorder:x,hasUnderlineHed:M,shouldHideDangerousDek:z,shouldHidePublishDate:!0,hasHedCorePrimary:!0,shouldPlayInline:K,hasNoBottomMarginForCneVideo:"cnevideo"===o,hasNoBottomPaddingForCneVideo:"cnevideo"===o}))}))(t);case"ArticleCarousel":return(e=>{const t=m[Y];return e.items.map((e,i)=>n.createElement(t,Object.assign({},e,{key:`${e.dangerousHed}-${i}`,hasBorder:x,hasUnderlineHed:M,shouldHideDangerousDek:z,shouldHidePublishDate:q})))})(t);default:return t.items.map((e,t)=>n.createElement(d,Object.assign({},e,{key:`${e.hed}-${t}`})))}}const me=D&&"GridFourColumns"===se,[he]=n.useState(parseInt(a(),10));return k&&0!==k.length?n.createElement(g,{dangerousHed:o,dangerousDek:v,editor:S,sectionTitleVariation:V,shouldEnableBundleComponentAnalytics:U,shouldUseAlternativeTitleStyle:J,hasImpressionTracking:!0,trackingNamespace:ie,isDotClickable:ne,hasCarouselSliderPagination:T,hasCarouselControl:de,hasControls:E,hasNavigationButtonVariation:L,hasToggleGridColor:me,hasPadding:I,layout:se,pos:he,paginationStyle:F,buttonConfig:t,gridConfig:ae,hasBorderBottom:$},k.map(e=>{const t=ue(e);Z&&e.url&&t.push(n.createElement(c,{aspectRatio:i,key:"cta-"+e.label,url:e.url},le(p.ctaMessage,{groupName:e.label.toLocaleLowerCase()})));const o=b(t);return n.createElement(s,{key:e.label,name:e.label,url:e.url},o)})):null};v.propTypes={buttonConfig:o.object,copilotId:o.string,ctaCardAspectRatio:o.arrayOf(o.number),dangerousDek:o.string,dangerousHed:o.string,dropShipSellers:o.arrayOf(o.string),editor:o.object,expVariationName:o.string,gridConfig:o.object,groups:o.arrayOf(o.object),hasAtRetailerNameLowerCase:o.bool,hasBorder:o.bool,hasBorderBottom:o.bool,hasCarouselSliderPagination:o.bool,hasControls:o.bool,hasGridColumn:o.bool,hasImageGrid:o.bool,hasMarginTopAuto:o.bool,hasNavigationButtonVariation:o.bool,hasPadding:o.bool,hasPLPBrandNameContextTitle:o.bool,hasPLPCardNameDescriptionCore:o.bool,hasProductNewPriceColor:o.bool,hasProductPriceColor:o.bool,hasProductWhiteBackground:o.bool,hasToggleGridColor:o.bool,hasUnderlineHed:o.bool,isDotClickable:o.bool,isDropshipProduct:o.bool,isProductCardName:o.bool,isProductCardRetailerName:o.bool,layout:o.string,paginationStyle:o.string,priceFormatting:o.shape({fractionDigits:o.number,shouldFormatDecimalSeparator:o.bool}),sectionTitleVariation:o.string,shouldCheckProductInView:o.bool,shouldEnableBundleComponentAnalytics:o.bool,shouldHideDangerousDek:o.bool,shouldHidePublishDate:o.bool,shouldPlayInline:o.bool,shouldRenderCtaCard:o.bool,shouldUseAlternativeTitleStyle:o.bool,shouldUseProductPriceSecondary:o.bool,showNewProductCardDesign:o.bool,showOfferUrl:o.bool,summaryItemVariation:o.string,trackingNamespace:o.shape({toggle:o.string,card:o.string})},v.displayName="VersoFilterableSummaryList",v.defaultProps={ctaCardAspectRatio:[2,1],dropShipSellers:[],hasAtRetailerNameLowerCase:!1,hasMarginTopAuto:!1,isDotClickable:!1,isDropshipProduct:!1,isProductCardRetailerName:!1,layout:"Carousel",sectionTitleVariation:"TextCenter",shouldEnableBundleComponentAnalytics:!1,shouldPlayInline:!1,shouldRenderCtaCard:!1,showNewProductCardDesign:!1,summaryItemVariation:"TextBelowLeft"},e.exports=v},1668:function(e,t,i){e.exports=i(1669)},1669:function(e,t,i){const n=i(0),o=i(1),{CTACardBody:a,CTACardContent:r,CTACardFooter:l,CTACardIcon:s,CTACardText:d,CTACardWrapper:c}=i(1670),p=({aspectRatio:e,children:t,url:i})=>n.createElement(c,{aspectRatio:e},n.createElement(r,null,n.createElement(a,null,n.createElement(d,{href:i},t)),n.createElement(l,null,n.createElement("a",{href:i},n.createElement(s,null)))));p.propTypes={aspectRatio:o.arrayOf(o.number),children:o.node.isRequired,url:o.string.isRequired},p.defaultProps={aspectRatio:[1,1]},e.exports=p},1670:function(e,t,i){const{default:n}=i(3),{BREAKPOINTS:o}=i(6),{calculateSpacing:a,getColorToken:r,getDecoration:l,minScreen:s}=i(4),{BaseLink:d}=i(10),c=i(123),p=n.div.withConfig({displayName:"CTACardWrapper"})`
  position: relative;
  border-radius: ${({theme:e})=>l(e,"cardRadiusSm")};
  background-color: ${r("colors.discovery.body.brand.background")};
  padding-top: ${({aspectRatio:e})=>e[1]/e[0]*100}%;

  ${s(o.md)} {
    border-radius: ${({theme:e})=>l(e,"cardRadiusMd")};
  }

  ${s(o.lg)} {
    border-radius: ${({theme:e})=>l(e,"cardRadiusLg")};
  }
`,g=n.div.withConfig({displayName:"CTACardContent"})`
  display: flex;
  position: absolute;
  top: 0;
  right: 0;
  bottom: 0;
  left: 0;
  flex-direction: column;
  padding: 0 ${a(3)} ${a(2)}
    ${a(3)};

  ${s(o.lg)} {
    padding: 0 ${a(5)} ${a(5)}
      ${a(5)};
  }
`,u=n.div.withConfig({displayName:"CTACardBody"})`
  display: flex;
  flex: 1 0 auto;
  align-items: center;
`,m=n.div.withConfig({displayName:"CTACardFooter"})`
  height: ${a(6)};
`,h=n(d).withConfig({displayName:"CTACardText"})``;h.defaultProps={colorToken:"colors.discovery.body.brand.heading",typeIdentity:"typography.definitions.discovery.hed-core-secondary"};const y=n(c).withConfig({displayName:"CTACardIcon"})`
  border-radius: 50%;
  background-color: ${r("colors.discovery.body.brand.context-signature")};
  width: ${a(6)};
  height: ${a(6)};
  fill: ${r("colors.discovery.body.brand.context-texture")};
`;e.exports={CTACardBody:u,CTACardContent:g,CTACardFooter:m,CTACardIcon:y,CTACardText:h,CTACardWrapper:p}},1671:function(e,t,i){const{defineMessages:n}=i(2);e.exports=n({ctaMessage:{id:"VersoFilterableSummaryList.CTAMessage",defaultMessage:"See more {groupName} recipes",description:"Message to display in the CTACard"}})},1672:function(e,t,i){const{default:n}=i(3),{BREAKPOINTS:o}=i(6),{minScreen:a,calculateSpacing:r,minMaxScreen:l}=i(4),{cssVariablesGrid:s,applyGridSpacing:d}=i(16),{LabelText:c,ListWrapper:p}=i(130),{CarouselListItem:g}=i(47),{CarouselWrapper:u,ChipWrapper:m}=i(1520),h=n(i(1618)).withConfig({displayName:"FilterableSummaryList"})`
  margin: 0 auto;
  ${({hasToggleGridColor:e})=>!e&&`max-width:${o.fullBleed};\n  ${s()}`}
  ${({hasPadding:e})=>!e&&d("padding")}
  ${g}:first-child {
    box-sizing: content-box;
    ${({hasPadding:e})=>e?`padding-left:${r(8)};\n      ${l(0,o.md)}\n      {  \n        padding-left:${r(3)};  \n      }`:"padding-left: 0;"}
  }
  ${g}:last-child {
    box-sizing: content-box;
    ${({hasPadding:e})=>e?`padding-right:${r(8)};\n    ${l(0,o.md)}\n     { padding-right:${r(3)}; \n    }`:null}
  }
  ${a(o.xxl)} {
    ${m},
    ${u} {
      ${({hasToggleGridColor:e})=>!e&&"margin-left: revert;\n      margin-right: revert;"}
      ${c} ,
      ${p} {
        padding-right: revert;
        padding-left: revert;
      }
      ${g}:first-child {
        ${({hasPadding:e})=>!e&&"padding-left: revert;"}
      }
      ${g}:last-child {
        ${({hasPadding:e})=>!e&&"padding-right: revert;"}
      }
    }
  }
`;e.exports={FilterableSummaryList:h}},1673:function(e,t){Object.defineProperty(t,"__esModule",{value:!0}),t.childTobeRendered=t.modifyItemObjectBasedOnContentType=void 0;t.modifyItemObjectBasedOnContentType=(e,t,i)=>("cnevideo"===t&&i&&(e.image=Object.assign(Object.assign({},e.image),{url:(null==e?void 0:e.url)||"",dangerousHed:e.dangerousHed||""})),e);t.childTobeRendered=e=>e.filter(e=>{var t,i,n,o,a,r,l;return((null===(i=null===(t=null==e?void 0:e.props)||void 0===t?void 0:t.image)||void 0===i?void 0:i.id)||(null===(n=null==e?void 0:e.props)||void 0===n?void 0:n.aspectRatio)||"cnevideo"===(null===(o=null==e?void 0:e.props)||void 0===o?void 0:o.contentType)&&((null===(r=null===(a=null==e?void 0:e.props)||void 0===a?void 0:a.image)||void 0===r?void 0:r.scriptUrl)||(null===(l=null==e?void 0:e.props)||void 0===l?void 0:l.url)))&&e})},1757:function(e,t){Object.defineProperty(t,"__esModule",{value:!0}),t.AGE_GATE_COOKIE_EXPIRATION_IN_DAYS=t.AGE_GATE_COOKIE_KEY=t.AGE_GATE_ACCEPT=void 0,t.AGE_GATE_ACCEPT="accept",t.AGE_GATE_COOKIE_KEY="ageGate",t.AGE_GATE_COOKIE_EXPIRATION_IN_DAYS=28},1758:function(e,t,i){e.exports=i(1759)},1759:function(e,t,i){const n=i(1),o=i(0),a=i(50),{PaywallInlineBarrierWrapper:r}=i(1760),{trackComponent:l}=i(5),s=function(e){o.useEffect(()=>{l("PaywallInlineBarrier")},[]);const{position:t,className:i}=e;return o.createElement(r,{className:i,"data-testid":"PaywallInlineBarrierWrapper"},o.createElement(a,{position:t,aria:{"aria-live":"polite"}}))};s.propTypes={className:n.string,position:n.string},s.defaultProps={position:"paywall-inline-barrier"},e.exports=s},1760:function(e,t,i){const{default:n}=i(3),o=n.aside.withConfig({displayName:"PaywallInlineBarrierWrapper"})`
  width: auto;
  height: auto;

  @media print {
    display: none;
  }
`;e.exports={PaywallInlineBarrierWrapper:o}},1761:function(e,t,i){const n=i(0),{useState:o}=i(0),a=i(1),r=i(170),{PaymentGateway:l}=i(23),{asConfiguredComponent:s}=i(9),{StickyMidContentAdWrapper:d}=i(265),c={"300x250":500,"300x100":400,"300x50":300,"320x50":300},p=e=>{const{isStickyEnabled:t}=Object.assign({},e),[i,a]=o(),s=e=>{a(e)},p=i&&2===i.length?`${i[0]}x${i[1]}`:"0x0";return t?n.createElement(d,{height:c[p],className:"ad-stickymidcontent"},n.createElement(l,{group:"ads"},n.createElement(r,Object.assign({position:"mid-content",handleAdSizeChange:s},e)))):n.createElement(l,{group:"ads"},n.createElement(r,Object.assign({position:"mid-content",handleAdSizeChange:s},e)))};p.propTypes={isStickyEnabled:a.bool},p.defaultProps={isStickyEnabled:!1},p.displayName="StickyMidContent",e.exports=s(p,"StickyMidContent")},1762:function(e,t,i){const n=i(1),o=i(0),{useIntl:a}=i(2),r=i(14),l=i(42),s=i(1763).default,{componentTracking:d}=i(13),{useOnAdFilled:c}=i(74),{asConfiguredComponent:p}=i(9),{trackComponent:g}=i(5),{SummaryCollectionSplitColumnsWrapper:u,SummaryCollectionSplitColumnsItems:m,SummaryCollectionSplitColumnsRecsWrapper:h,SummaryCollectionSplitColumnsItem:y}=i(1764),b=({className:e,dangerousHed:t,guideItem:i,hasBackgroundColor:n,hasBorderItem:l,hasExtraRubricSpace:p,hasLessBottomSpace:b,hasTighterBylineSpacing:f,itemHedTag:C,location:v,recommendedItems:w,recommendedItemCount:S,shouldHideDangerousDek:k,shouldAppendReadMoreLinkForDek:x,shouldHideBylines:$,shouldEnableBundleComponentAnalytics:T,summaryItemRubricVariation:E,trackingNamespace:N})=>{o.useEffect(()=>{g("SummaryCollectionSplitColumns")},[]);const{formatMessage:B}=a(),{items:L,recommendedType:I,recommendedClickout:A}=w,P=C||(t?"h3":"h2"),[R]=c("trending-ad");return o.createElement(u,{className:e,"data-testid":"SummaryCollectionSplitColumnsWrapper",hasBackgroundColor:n},o.createElement(m,{"data-testid":"SummaryCollectionSplitColumnsItems",showTrendingAd:R},o.createElement(h,null,o.createElement("p",null,B(s.recommendedTitle,{recommendedType:I})),L.slice(0,S).map((e,t)=>{const i=d.addDataSectionTitleAttribute(T,null==N?void 0:N.item,t);return o.createElement(o.Fragment,{key:t},o.createElement(y,Object.assign({},e,{analyticsDataAttribute:i,variation:"TextBelowImageLeftHasRuleWithDek",hasBorder:l,hedTag:P,key:t,rubricVariation:E,"data-testid":"SummaryCollectionSplitColumnsItem",shouldHideDangerousDek:k,shouldAppendReadMoreLinkForDek:x,shouldHideBylines:$,hasLessBottomSpace:b,hasExtraRubricSpace:p})))}),A&&o.createElement(r.Utility,{label:B(s.viewAllButton,{location:v,recommendedType:I}),inputKind:"link",href:A})),o.createElement(y,Object.assign({},i[0],{image:i[0].lede,dangerousHed:v?B(s.guideItemHed,{location:v}):t,variation:"TextBelowImageLeftHedAndDek",hasBorder:l,hedTag:P,rubricVariation:E,"data-testid":"SummaryCollectionSplitColumnsItem",shouldHideBylines:$,hasTighterBylineSpacing:f,hasLessBottomSpace:b,hasExtraRubricSpace:p}))))};b.propTypes={className:n.string,dangerousHed:n.string,guideItem:n.arrayOf(n.shape(l.propTypes)).isRequired,hasBackgroundColor:n.bool,hasBorderItem:n.bool,hasExtraRubricSpace:n.bool,hasLessBottomSpace:n.bool,hasTighterBylineSpacing:n.bool,itemHedTag:n.string,location:n.string,recommendedItemCount:n.number,recommendedItems:n.shape({items:n.arrayOf(n.shape(l.propTypes)),recommendedType:n.string,recommendedClickout:n.string}).isRequired,shouldAppendReadMoreLinkForDek:n.bool,shouldEnableBundleComponentAnalytics:n.bool,shouldHideBylines:n.bool,shouldHideDangerousDek:n.bool,summaryItemRubricVariation:n.string,trackingNamespace:n.object},b.defaultProps={hasBackgroundColor:!0,hasBorderItem:!0,hasExtraRubricSpace:!1,hasLessBottomSpace:!1,hasTighterBylineSpacing:!1,recommendedItemCount:2,shouldAppendReadMoreLinkForDek:!0,shouldHideBylines:!0,shouldHideDangerousDek:!1},b.displayName="SummaryCollectionSplitColumns",e.exports=p(b,"SummaryCollectionSplitColumns")},1763:function(e,t,i){Object.defineProperty(t,"__esModule",{value:!0});const n=i(2);t.default=n.defineMessages({viewAllButton:{id:"SummaryCollectionSplitColumns.ViewAllButton",defaultMessage:"View All {location} {recommendedType}",description:"button label for view all button"},guideItemHed:{id:"SummaryCollectionSplitColumns.GuideItemHed",defaultMessage:"{location} Travel Guide",description:"dangerousHed for guideItem"},recommendedTitle:{id:"SummaryCollectionSplitColumns.RecommendedTitle",defaultMessage:"Recommended {recommendedType}",description:"recommended title for recircs"}})},1764:function(e,t,i){var n=this&&this.__rest||function(e,t){var i={};for(var n in e)Object.prototype.hasOwnProperty.call(e,n)&&t.indexOf(n)<0&&(i[n]=e[n]);if(null!=e&&"function"==typeof Object.getOwnPropertySymbols){var o=0;for(n=Object.getOwnPropertySymbols(e);o<n.length;o++)t.indexOf(n[o])<0&&Object.prototype.propertyIsEnumerable.call(e,n[o])&&(i[n[o]]=e[n[o]])}return i};const o=i(0),{default:a}=i(3),{BylineWrapper:r,BylinePreamble:l,BylineName:s,BylineLink:d}=i(103),c=i(42),{GridItem:p}=i(25),g=i(19),{calculateSpacing:u,getColorToken:m,getColorStyles:h,getTypographyStyles:y,minScreen:b,styledProperty:f}=i(4),{BREAKPOINTS:C}=i(6),v=a.div.withConfig({displayName:"SummaryCollectionSplitColumnsWrapper"})`
  margin: ${u(4)} 0;
  background-color: ${({hasBackgroundColor:e})=>e?m("colors.discovery.body.light.background"):"transparent"};
`,w=`\n  &:last-child {\n    padding-bottom: 0;\n    \n    ${b(C.md)}{\n      padding-bottom: ${u(2)};\n    }\n  }\n`,S=a(e=>{var{columnAmount:t,variation:i,shouldHideDangerousDek:a,hasExtraRubricSpace:r,hasLessBottomSpace:l,hasTighterBylineSpacing:s}=e,d=n(e,["columnAmount","variation","shouldHideDangerousDek","hasExtraRubricSpace","hasLessBottomSpace","hasTighterBylineSpacing"]);const p=c[i];return o.createElement(p,Object.assign({},d))}).withConfig({displayName:"SummaryCollectionSplitColumnsItem"})`
  ${v} & {
    padding-bottom: ${u(2)};

    ${b(C.md)} {
      border-bottom: 0;
    }

    .summary-item__rubric {
      ${y("typography.definitions.globalEditorial.context-primary")}
      display: block;
      color: ${m("colors.discovery.body.light.context-signature")};

      ${b(C.md)} {
        margin-bottom: ${({hasExtraRubricSpace:e})=>u(e?1:.5)};
      }
    }

    .summary-item__hed-link {
      color: ${m("colors.discovery.body.light.heading")};

      &::after {
        display: none;
      }
    }

    .summary-item__hed {
      ${y("typography.definitions.discovery.hed-bulletin-primary")}
      margin-bottom: 0;
    }

    .summary-item__dek {
      ${y("typography.definitions.discovery.description-page")}
      display: ${({shouldHideDangerousDek:e})=>e?"none":"block"};
      margin: ${u(2)} 0 ${u(1)};
      color: ${m("colors.discovery.body.white.description")};
    }

    .summary-item__content {
      padding-bottom: ${({hasLessBottomSpace:e})=>e?u(0):""};
    }

    .summary-item__byline {
      margin-top: ${({hasTighterBylineSpacing:e})=>u(e?1:2)};

      ${r},
      ${l},
      ${s},
      ${d} {
        ${y("typography.definitions.globalEditorial.accreditation-core")}
        color: ${m("colors.discovery.body.light.accreditation")};
      }

      ${d}:link {
        color: ${m("colors.discovery.body.light.accreditation")};
      }
    }

    .summary-item__metadata-secondary {
      margin: ${u(2)} 0 0 0;
    }

    ${f("hasBorder",!1,w)};
  }
`,k=a(g.ThreeUp).withConfig({displayName:"SummaryCollectionSplitColumnsItems"})`
  ${p} {
    grid-column: 1 / -1;
    margin-top: ${u(2)};
  }
  ${p}:first-of-type {
    margin: ${u(3)} 0;
    ${b(C.md)} {
      grid-column: span 7;
    }
  }
  ${p}:last-of-type {
    ${b(C.md)} {
      grid-column: span 5;
      margin: ${u(3)} 0;
    }
  }
`,x=a("div").withConfig({displayName:"SummaryCollectionSplitColumnsRecsWrapper"})`
  display: grid;
  grid-column-gap: ${u(3)};
  grid-template-columns: repeat(6, 1fr);
  grid-row-gap: ${u(2)};
  height: 100%;

  ${b(C.md)} {
    display: grid;
    grid-template-rows: 5% 75% 20%;
    grid-row-gap: ${u(1)};
    border-right: 1px solid;
    border-color: ${({theme:e})=>h(e,"border-color","colors.consumption.body.standard.divider")};
    padding-right: ${u(3)};
  }

  p {
    grid-column: span 6;
    grid-row: 1 / 1;
    align-self: center;
    margin: 0;
    ${({theme:e})=>y(e,"typography.definitions.foundation.link-primary")}
  }

  .summary-item {
    display: grid;
    grid-column-gap: 1rem;
    grid-column: span 6;
    grid-template-columns: 40% 60%;
    align-items: center;

    ${b(C.sm)} {
      grid-template-columns: 1fr 1fr;
    }

    ${b(C.md)} {
      display: unset;
      grid-column: span 3;
      grid-row: 2 / 2;
    }
  }

  .button {
    grid-column: span 6;
    max-height: ${u(6)};
    ${b(C.md)} {
      grid-column: 2 / span 4;
      grid-row: 3;
    }
    ${b(C.lg)} {
      justify-self: center;
      padding: 0 ${u(6)};
    }
  }
`;e.exports={SummaryCollectionSplitColumnsWrapper:v,SummaryCollectionSplitColumnsItems:k,SummaryCollectionSplitColumnsRecsWrapper:x,SummaryCollectionSplitColumnsItem:S}},1765:function(e,t,i){Object.defineProperty(t,"__esModule",{value:!0}),t.GalleryRecircViewGalleryCTA=t.GalleryRecircTitle=t.GalleryRecircImage=t.GalleryRecircHeading=t.GalleryMidRecircHeading=t.GalleryRecircGridWrapper=t.GalleryRecircContent=void 0;const{css:n,default:o}=i(3),a=i(49),{calculateSpacing:r,getColorToken:l,maxScreen:s}=i(4),{maxThresholds:d}=i(17),{BaseWrap:c,BaseText:p}=i(10),g=o.div.withConfig({displayName:"GalleryRecircGridWrapper"})`
  display: grid;
  grid-template-rows: repeat(3, auto);
  grid-row-gap: ${r(2)};
  margin: auto;
  max-width: ${r(54)};
  height: auto;

  ${s(d.lg+"px")} {
    max-width: unset;
  }

  ${({isEndOfPageRecirc:e})=>e&&n`
      grid-template-rows: repeat(1, 1fr);
      grid-row-gap: 0;
      justify-content: end;
      max-width: unset;

      ${s(d.lg+"px")} {
        grid-template-columns: repeat(1, 1fr);
      }
    `}
`;t.GalleryRecircGridWrapper=g;const u=o(c).withConfig({displayName:"GalleryRecircContent"})`
  display: grid;
  grid-template-columns: repeat(12, 1fr);
  grid-template-rows: repeat(2, 1fr);
  grid-row-gap: ${r(1)};
  border: 1px solid ${l("colors.consumption.body.standard.divider")};
  border-radius: ${r(2)};
  cursor: pointer;
  padding: ${r(2)} ${r(3)};

  ${({isEndOfPageRecirc:e})=>e&&n`
      grid-gap: 0;
      grid-template-rows: repeat(4, auto);
      border-right: none;
      border-top-right-radius: 0;
      border-bottom-right-radius: 0;
      padding: ${r(3)} ${r(4)};
      max-width: ${r(29)};
      height: auto;

      ${s(d.lg+"px")} {
        grid-template-rows: repeat(3, auto);
        border-right: 1px solid
          ${l("colors.consumption.body.standard.divider")};
        border-bottom: none;

        border-top-right-radius: ${r(2)};
        border-bottom-left-radius: 0;
        border-bottom-right-radius: 0;
        padding-right: ${r(2)};
        padding-left: ${r(2)};
        max-width: 100%;
        justify-items: center;
      }
    `}
`;t.GalleryRecircContent=u,u.defaultProps={as:"a"};const m=o(p).withConfig({displayName:"GalleryRecircTitle"})`
  display: flex;
  grid-column: 1/9;
  grid-row: 1/1;
  align-items: flex-end;
  padding-right: ${r(2)};

  ${({isEndOfPageRecirc:e})=>e&&n`
      grid-column: 1/-1;
      grid-row: 3;
      margin-bottom: ${r(1)};

      ${s(d.lg+"px")} {
        grid-column: 1/9;
        grid-row: 2;
        text-align: center;
      }
    `}
`;t.GalleryRecircTitle=m,m.defaultProps={colorToken:"colors.consumption.body.standard.subhed",typeIdentity:"typography.definitions.globalEditorial.context-primary"};const h=o(p).withConfig({displayName:"GalleryRecircViewGalleryCTA"})`
  grid-column: 1/9;
  grid-row: 2;

  :hover {
    text-decoration: underline;
  }

  ${({isEndOfPageRecirc:e})=>e&&n`
      grid-column: 1/-1;
      grid-row: 4;

      ${s(d.lg+"px")} {
        grid-column: 1/8;
        grid-row: 3;
      }
    `}
`;t.GalleryRecircViewGalleryCTA=h,h.defaultProps={as:"span",colorToken:"colors.interactive.base.brand-primary",typeIdentity:"typography.definitions.foundation.link-primary"};const y=o(a).withConfig({displayName:"GalleryRecircImage"})`
  display: flex;
  grid-column: 9/-1;
  grid-row: 1/3;
  justify-content: flex-end;

  img {
    max-width: ${r(15.5)};

    ${s(d.lg+"px")} {
      max-width: ${r(11.25)};
    }
  }

  ${({isEndOfPageRecirc:e})=>e&&n`
      grid-column: 1/-1;
      grid-row: 2;
      margin-bottom: ${r(2)};

      img {
        max-width: ${r(21)};
      }

      ${s(d.lg+"px")} {
        grid-column: 8/-1;
        grid-row: 1/-1;
        margin: 0;

        img {
          max-width: ${r(15.5)};
        }
      }
    `}
`;t.GalleryRecircImage=y;const b=o(p).withConfig({displayName:"GalleryRecircHeading"})`
  grid-column: 1/-1;
  grid-row: 1;
  padding-bottom: ${r(2)};

  ${s(d.lg+"px")} {
    grid-column: 1/8;
    padding-bottom: ${r(1)};
  }
`;t.GalleryRecircHeading=b,b.defaultProps={as:"h4",colorToken:"colors.consumption.body.standard.display-texture",typeIdentity:"typography.definitions.globalEditorial.context-secondary"};const f=o(p).withConfig({displayName:"GalleryMidRecircHeading"})`
  grid-column: 1/-1;
  grid-row: 1;
  text-align: center;

  ${s(d.lg+"px")} {
    padding: 0 ${r(6)};
  }
`;t.GalleryMidRecircHeading=f,f.defaultProps={as:"h4",colorToken:"colors.consumption.body.standard.display-texture",typeIdentity:"typography.definitions.globalEditorial.context-secondary"}},179:function(e,t,i){const n=i(8),o=i(1),a=i(0),{useIntl:r}=i(2),l=i(89),{storyVideoPosition:s}=i(234),d=i(231),c=i(235),p=i(11),g=i(21),{transformLegacySources:u}=i(93),m=i(164).default,{ContentHeaderLeadAsset:h,ContentHeaderResponsiveAsset:y,ContentHeaderLeadAssetContent:b,ContentHeaderLeadAssetCaption:f,ContentHeaderLeadAssetContentMedia:C,ContentHeaderLedeLightboxButton:v,ContentHeaderLeadRailAnchor:w,ContentHeaderLeadContentFullWidth:S,ContentHeaderLeadContentCaptionCredit:k,ContentHeaderLeadAssetAwards:x}=i(102),{useState:$,Fragment:T}=a,E="landscape",N="portrait",B=({awards:e,className:t,captionWidth:i,containerTheme:o,lede:p,mediaWidth:B,shouldRenderRailAnchor:L,showFullWidthLeadImage:I,socialIcons:A,hasLightboxButton:P,hasStaticPositionedAward:R,hasInvertedLedeBackground:D,hideLedeCaption:M})=>{const[O,H]=$((e=>{var t;const i=null===(t=null==e?void 0:e.masterAspectRatio)||void 0===t?void 0:t.split(":");return(null==e?void 0:e.restrictCropping)&&2===(null==i?void 0:i.length)&&Number(i[0])/Number(i[1])<=1?N:E})(p)),{formatMessage:_}=r(),G=({width:e,height:t})=>{e/t<=1&&H(N)},W=u(p),j="cnevideo"===p.modelName,F="gallery"===p.modelName,V="clip"===p.modelName,U=!M&&(p.caption&&p.caption.trim()||p.credit&&p.credit.trim()),z=a.createElement(f,{dangerousCaptionText:p.caption,dangerousCredit:p.credit,mediaWidth:B});return a.createElement(T,null,a.createElement(h,{className:n("lead-asset",t),mediaWidth:B,containerTheme:o,ledeContentType:p.contentType,ledeAssetOrientation:O,hasInvertedLedeBackground:D,"data-testid":"ContentHeaderLeadAsset"},a.createElement(b,null,a.createElement(C,{ledeContentType:p.contentType,showFullWidthLeadImage:I,mediaWidth:B,className:"lead-asset__content__"+p.contentType},!j&&!F&&!V&&a.createElement(T,null,e&&!R&&a.createElement(x,{awards:e,hasStaticPositionedAward:R}),a.createElement(y,Object.assign({},W,{onAssetLoaded:G,shouldRestrictCropping:null==p?void 0:p.restrictCropping,masterAspectRatio:null==p?void 0:p.masterAspectRatio,shouldHoldImageSpace:!0,mediaWidth:B,"data-testid":"assetMedia"}))),j&&p.scriptEmbedUrl&&a.createElement(l,{shouldAutoplay:!0,scriptUrl:p.scriptEmbedUrl,shouldHaveTeaser:!0,videoEmbedPosition:s}),F&&a.createElement(d,Object.assign({},p,{showNoAdsFromParent:!0})),A&&a.createElement(g.Overlay,{links:A.links}),V&&a.createElement(y,Object.assign({},W,{onAssetLoaded:G,shouldRestrictCropping:null==p?void 0:p.restrictCropping,masterAspectRatio:null==p?void 0:p.masterAspectRatio,shouldHoldImageSpace:!0,mediaWidth:B,"data-testid":"assetMedia"})),P&&a.createElement(v,{onClickHandler:()=>{document.querySelector(".responsive-image--expandable").click()},ButtonIcon:()=>a.createElement(c,null),hasEnableIcon:!0,btnStyle:"text",iconPosition:"before",inputKind:"button",isStaticText:!0,label:_(m.showAllPhotos),shouldRenderCaption:U}),U&&"fullbleed"!==i&&z,e&&R&&a.createElement(x,{awards:e,hasStaticPositionedAward:R}))),L&&a.createElement(w,{"data-testid":"ContentHeaderLeadRailAnchor"})),U&&"fullbleed"===i&&a.createElement(S,null,a.createElement(k,null,z)))};B.propTypes={awards:o.array,captionWidth:o.oneOf(["standard","fullbleed"]),className:o.string,containerTheme:o.oneOf(["standard","inverted","special"]),hasInvertedLedeBackground:o.bool,hasLightboxButton:o.bool,hasStaticPositionedAward:o.bool,hideLedeCaption:o.bool,lede:o.oneOfType([o.shape(p.propTypes),o.shape(l.propTypes)]).isRequired,mediaWidth:o.oneOf(["small","smallrule","grid","fullbleed"]),shouldRenderRailAnchor:o.bool,showFullWidthLeadImage:o.bool,socialIcons:o.shape(g.propTypes)},B.defaultProps={captionWidth:"standard",hasStaticPositionedAward:!1,hideLedeCaption:!1},e.exports=B,e.exports.transformLegacySources=u},180:function(e,t,i){const n=i(3).default,{BaseLink:o,BaseText:a}=i(10),{calculateSpacing:r,getLinkStyles:l,getTypographyStyles:s,getColorStyles:d,getInputFieldStyles:c}=i(4),p=i(33),g=n.div.withConfig({displayName:"GroupedNavigationWrapper"})`
  padding-top: ${r(4)};
  height: 100%;

  ${({hasFilter:e})=>e&&`padding-top: ${r(2)};`}

  .navigation__heading {
    ${({theme:e})=>s(e,"typography.definitions.foundation.title-primary")};
    margin: 0;
    line-height: normal;
    ${({theme:e})=>d(e,"color","colors.foundation.expanded-utility.nav-link.default")};
  }

  .navigation__list-item {
    white-space: normal;
  }

  .content-divider {
    display: block;
    margin-bottom: ${r(2)};
    border-bottom-width: ${r(.5)};
    border-bottom-style: solid;
    ${({theme:e})=>d(e,"border-bottom-color","colors.discovery.lead.secondary.accent")};
    padding-top: ${r(1)};
    width: ${r(2)};
  }

  .grouped-navigation__link {
    ${({theme:e})=>l(e,"colors.foundation.expanded-utility.nav-link.default","colors.foundation.expanded-utility.nav-link.hover","navigation")}

    &.link--primary {
      ${({theme:e})=>s(e,"typography.definitions.foundation.link-primary")};
    }

    &.link--secondary {
      ${({theme:e})=>s(e,"typography.definitions.foundation.link-secondary")};
    }
  }
`,u=n.div.withConfig({displayName:"GroupedNavigationFilter"})`
  position: static;
  border-width: 0 0 1px;
  border-style: solid;
  ${({theme:e})=>d(e,"color","colors.discovery.body.white.divider")};
  width: calc(100% - 1.25rem);
  height: 60px;

  .icon {
    position: absolute;
    top: 10px;
    right: 0;
    pointer-events: none;
  }
`,m=n.div.withConfig({displayName:"GroupedNavigationFilterContent"})`
  position: relative;
`,h=n(a).withConfig({displayName:"GroupedNavigationFilterInput"})`
  ${({theme:e})=>c(e,"normal","background")};
  ${({theme:e})=>c(e,"normal","text")};
  border: none;
  width: 100%;
  height: ${r(6.2)};
`;h.defaultProps={as:"input",typeIdentity:"typography.definitions.foundation.link-secondary"};const y=n.div.withConfig({displayName:"GroupedNavigationContent"})`
  display: flex;
  height: 100%
    ${({hasFilter:e})=>e&&`\n      padding-top: ${r(4)};\n      height: calc(100% - 60px);\n  `};
`,b=n.div.withConfig({displayName:"GroupedNavigationLinks"})`
  flex: 1;
  height: 100%;
  overflow-y: auto;
  max-height: 100vh;

  && li {
    padding-bottom: ${r(2)};

    &.link--primary {
      ${({theme:e})=>s(e,"typography.definitions.foundation.link-primary")};
    }

    &.link--secondary {
      ${({theme:e})=>s(e,"typography.definitions.foundation.link-secondary")};
    }
  }
`,f=n(p.Vertical).withConfig({displayName:"GroupedNavigationGroup"})`
  padding-bottom: ${r(5)};
`,C=n.div.withConfig({displayName:"GroupedNavigationIndex"})`
  position: static;
  padding-right: ${r(1)};
  overflow-y: auto;
`,v=n(a).withConfig({displayName:"AtoZIndexWrapper"})`
  width: ${r(3)};
  text-align: center;
`;v.defaultProps={as:"nav"};const w=n(a).withConfig({displayName:"AtoZIndexList"})`
  margin-top: 0;
  padding: 0;
  list-style: none;
`;w.defaultProps={as:"ul"};const S=n(o).withConfig({displayName:"AtoZIndexLink"})`
  display: block;
  background: none;
  padding-top: ${r(.25)};
  padding-bottom: ${r(.25)};
  width: 100%;
`;S.defaultProps={colorSecondaryLinkToken:"colors.foundation.expanded-utility.nav-link.hover",colorStaticLinkToken:"colors.foundation.expanded-utility.nav-link.default",colorToken:"colors.foundation.expanded-utility.nav-link.default",linkStyle:"navigation",typeToken:"typography.definitions.foundation.link-secondary"};const k=n(a).withConfig({displayName:"AtoZIndexText"})`
  margin: 0;
  padding-top: ${r(.25)};
  padding-bottom: ${r(.25)};
`;k.defaultProps={as:"li",colorToken:"colors.foundation.expanded-utility.nav-link.default",typeIdentity:"typography.definitions.foundation.link-secondary"},e.exports={GroupedNavigationWrapper:g,GroupedNavigationFilter:u,GroupedNavigationFilterContent:m,GroupedNavigationFilterInput:h,GroupedNavigationContent:y,GroupedNavigationLinks:b,GroupedNavigationGroup:f,GroupedNavigationIndex:C,AtoZIndexWrapper:v,AtoZIndexList:w,AtoZIndexLink:S,AtoZIndexText:k}},1940:function(e,t,i){const{asConfiguredComponent:n}=i(9),o=i(1941);e.exports=n(o,"AgeGate")},1941:function(e,t,i){const n=i(0),o=i(1),{useIntl:a}=i(2),{trackComponent:r}=i(5),{HiddenCheckbox:l,Overlay:s,Title:d,Text:c,Logo:p,DefaultLogo:g,AgeGateButton:u}=i(1942),{AGE_GATE_ACCEPT:m,AGE_GATE_COOKIE_KEY:h}=i(1757),{hasContentWarnings:y,acceptAgeGatePrompt:b}=i(1943),{getCookie:f}=i(38),C=i(1944).default,v=({hed:e,dek:t,acceptLabel:i,declineLabel:o,logo:v,content:w,cookieExpirationInDays:S,brandContentWarnings:k})=>{n.useEffect(()=>{r("AgeGate")},[]);const{useState:x,useEffect:$}=n,{formatMessage:T}=a(),[E,N]=x(!1);if($(()=>{const e=!(f(h)===m)&&y({content:w,brandContentWarnings:k});N(e)},[w,k]),!E)return null;const B=null!=t?t:T(C.ageGateDekText);return n.createElement(n.Fragment,null,n.createElement(l,{id:"age-gate-check"}),n.createElement(s,{id:"age-gate-overlay",role:"dialog","aria-labelledby":"age-gate-title","aria-describedby":"age-gate-description"},v?n.createElement(p,{src:v,alt:e}):n.createElement(g,{width:96,height:96}),n.createElement(d,{as:"h2",id:"age-gate-title"},null!=e?e:T(C.ageGateHedText)),B&&n.createElement(c,{id:"age-gate-description"},B),n.createElement("label",{htmlFor:"age-gate-check",key:"age-gate-label-accept"},n.createElement(u,{inputKind:"link",onClickHandler:()=>((e,t)=>{e(!1),b(t)})(N,S),key:"age-gate-button-accept",dataAttrs:{"data-test-id":"age-gate-button-accept"},label:i||T(C.ageGateAcceptLabel)})),n.createElement(u,{href:"/",inputKind:"link",key:"age-gate-button-decline",dataAttrs:{"data-test-id":"age-gate-button-decline"},label:o||T(C.ageGateDeclineLabel)})))};v.displayName="AgeGate",v.propTypes={acceptLabel:o.string,brandContentWarnings:o.array,content:o.object.isRequired,cookieExpirationInDays:o.number,declineLabel:o.string,dek:o.string,hed:o.string,logo:o.string},v.defaultProps={brandContentWarnings:["sexual_content","drug_content","death_content","alcohol_content"]},e.exports=v},1942:function(e,t,i){var n=this&&this.__importDefault||function(e){return e&&e.__esModule?e:{default:e}};Object.defineProperty(t,"__esModule",{value:!0}),t.AgeGateButton=t.Text=t.Title=t.Logo=t.DefaultLogo=t.Overlay=t.HiddenCheckbox=void 0;const o=n(i(3)),a=i(17),r=i(4),l=n(i(111)),s=i(10),d=n(i(14));t.HiddenCheckbox=o.default.input.withConfig({displayName:"AgeGateCheckbox"})``,t.HiddenCheckbox.defaultProps={hidden:!0,type:"checkbox"},t.Overlay=o.default.div.withConfig({displayName:"AgeGateOverlay"})`
  display: flex;
  position: fixed;
  top: 0;
  left: 0;
  flex-direction: column;
  flex-wrap: wrap;
  align-items: center;
  justify-content: center;
  z-index: 10000;
  background: ${r.getColorToken("colors.consumption.lead.inverted.background")};
  padding: 0 1rem;
  width: 100%;
  height: 100%;
  text-align: center;

  ${t.HiddenCheckbox}:checked ~ & {
    display: none;
  }

  @media (min-width: ${a.minThresholds.lg}px) {
    padding: 0 10rem;
  }

  @media (min-width: ${a.minThresholds.xl}px) {
    padding: 0 20rem;
  }
`,t.DefaultLogo=o.default(l.default.AgeGate).withConfig({displayName:"AgeGateDefaultLogo"})`
  margin: 0 0 ${r.calculateSpacing(3)};
  fill: ${({theme:e})=>r.getColorToken(e,"colors.consumption.lead.inverted.heading")};
  width: 96px;
  height: 96px;

  path:first-of-type {
    fill: ${({theme:e})=>r.getColorToken(e,"colors.consumption.lead.inverted.accent")};
  }
`,t.Logo=o.default.img.withConfig({displayName:"AgeGateLogo"})`
  margin: 0 0 ${r.calculateSpacing(3)};
  width: 96px;
  height: 96px;
`,t.Title=o.default(s.BaseText).withConfig({displayName:"AgeGateTitle"})`
  margin: 0 0 ${r.calculateSpacing(2)};

  & + label,
  & + button {
    margin-top: ${r.calculateSpacing(2)};
  }
`,t.Title.defaultProps={colorToken:"colors.consumption.lead.inverted.heading",typeIdentity:"typography.definitions.consumptionEditorial.hed-bulletin"},t.Text=o.default(s.BaseText).withConfig({displayName:"AgeGateText"})`
  margin: 0 0 ${r.calculateSpacing(4)};
`,t.Text.defaultProps={colorToken:"colors.consumption.lead.inverted.heading",typeIdentity:"typography.definitions.consumptionEditorial.description-core"},t.AgeGateButton=o.default(d.default.Primary).withConfig({as:"a",displayName:"AgeGateButton"})`
  margin: 0 0 ${r.calculateSpacing(2)};
`},1943:function(e,t,i){Object.defineProperty(t,"__esModule",{value:!0}),t.acceptAgeGatePrompt=t.hasContentWarnings=void 0;const n=i(1757),{createCookie:o}=i(38);t.hasContentWarnings=({content:e,brandContentWarnings:t}={})=>{const{contentWarnings:i}=e||{},n=null==t?void 0:t.some(e=>null==i?void 0:i.some(t=>t.slug===e));return Boolean(n)};t.acceptAgeGatePrompt=e=>{document.cookie=o(n.AGE_GATE_COOKIE_KEY,n.AGE_GATE_ACCEPT,{expirationInMs:864e5*(e||n.AGE_GATE_COOKIE_EXPIRATION_IN_DAYS),path:"/"})}},1944:function(e,t,i){Object.defineProperty(t,"__esModule",{value:!0});const n=i(2);t.default=n.defineMessages({ageGateHedText:{id:"AgeGate.HedText",defaultMessage:"Are you 18 or over?",description:"Age Gate title"},ageGateDekText:{id:"AgeGate.DekText",defaultMessage:"This material is intended for people over the age of 18",description:"Age Gate description"},ageGateAcceptLabel:{id:"AgeGate.AcceptLabel",defaultMessage:"I am 18+",description:"Age Gate accept button label"},ageGateDeclineLabel:{id:"AgeGate.DeclineLabel",defaultMessage:"I'm under 18",description:"Age Gate decline button label"}})},1945:function(e,t,i){const{asConfiguredComponent:n}=i(9),{asVariation:o}=i(12),a=i(1946);a.WithAdRail=o(a,"WithAdRail",{adRail:!0}),a.OneColumn=o(a,"OneColumn",{adRail:!1}),a.OneColumnNarrow=o(a,"OneColumnNarrow",{adRail:!1,isNarrowContentWell:!0}),e.exports=n(a,"ChunkedArticleContent")},1946:function(e,t,i){const n=i(1),o=i(0),a=i(8),r=i(463),{getVariationNames:l}=i(12),s=i(19),{PaymentGateway:d}=i(23),c=i(58),p=i(1758),g=i(151),u=i(1947),{ArticlePageDisclaimer:m,ArticlePageChunksContent:h,ArticlePageChunks:y,ArticlePageDisclaimerGrid:b,PaywallInlineBarrierWithWrapperGrid:f}=i(1563),C=({body:e,hasProduct:t,linkList:i,isLinkStackEnabled:n,hasProductLisitingCarousel:r,isAffiliateLinksDisabled:l,hasTopSpacing:g,horizontalRuleStyle:C,interlude:v,multiChunkRailStrategy:w,shouldBlurText:S,shouldUsePersistentAd:k,singleChunkRailStrategy:x,variations:$,hideRecircMostPopular:T,dividerColor:E,shouldEnableArticleBackground:N,pageBackgroundTheme:B,recircMostPopular:L,showFirstRailRecirc:I,hideAutoDisclaimer:A,wordsToDisplay:P,hideAffiliateDisclaimer:R,hideProductDisclaimer:D,responsiveCartoonVariation:M,hasCartoonFullWidth:O,setCartoonLinkedGalleries:H})=>{const _=$&&$.adRail,G=a("article__body",{"article__body--grid-margins":!_}),W=_?s.NarrowContentWithWideAdRail:c,j=_?s.NarrowContentWithWideAdRail:s.WithMargins;return o.createElement(h,{isNarrowContentWell:$.isNarrowContentWell,shouldBlurText:S},(t||r)&&!l&&!A&&!R&&D&&o.createElement(b,{adRail:_,as:j},o.createElement(m,null)),o.createElement(y,{hasTopSpacing:g,horizontalRuleStyle:C,pageBackgroundTheme:B,"data-testid":"ArticlePageChunks"},o.createElement(u,{jsonml:e,adRail:_,linkList:i,isLinkStackEnabled:n,multiChunkRailStrategy:w,interlude:v,FullBleedContentWrapper:W,GeneralContentWrapper:j,recircMostPopular:L,shouldUsePersistentAd:k,singleChunkRailStrategy:x,hideRecircMostPopular:T,pageBackgroundTheme:B,dividerColor:E,shouldEnableArticleBackground:N,showFirstRailRecirc:I,wordsToDisplay:P,hideAffiliateDisclaimer:R,responsiveCartoonVariation:M,hasCartoonFullWidth:O,setCartoonLinkedGalleries:H})),o.createElement(d,{group:"paywall"},o.createElement(e=>o.createElement(f,{adRail:_,as:j},o.createElement("div",{className:a("body","body__inline-barrier",G)},o.createElement("div",{className:"container container--body"},o.createElement("div",{className:"container--body-inner"},o.createElement(p,Object.assign({},e)))))),null)))};C.propTypes={body:n.array.isRequired,dividerColor:n.string,hasCartoonFullWidth:n.bool,hasProduct:n.bool,hasProductLisitingCarousel:n.bool,hasTopSpacing:n.bool,hideAffiliateDisclaimer:n.bool,hideAutoDisclaimer:n.bool,hideProductDisclaimer:n.bool,hideRecircMostPopular:n.bool,horizontalRuleStyle:n.oneOf(["thin"]),interlude:n.shape(Object.assign(Object.assign({},g.propTypes),{isRailEligible:n.boolean})),isAffiliateLinksDisabled:n.bool,isLinkStackEnabled:n.bool,linkList:n.object,multiChunkRailStrategy:n.oneOf(["alternate"]),pageBackgroundTheme:n.string,recircMostPopular:n.array,responsiveCartoonVariation:n.oneOf(l(r)),setCartoonLinkedGalleries:n.func,shouldBlurText:n.bool,shouldEnableArticleBackground:n.bool,shouldUsePersistentAd:n.bool,showFirstRailRecirc:n.bool,singleChunkRailStrategy:n.oneOf(["split-in-three"]),variations:n.shape({adRail:n.bool,isNarrowContentWell:n.bool}),wordsToDisplay:n.number},C.defaultProps={hasCartoonFullWidth:!1,hasTopSpacing:!0,hideAutoDisclaimer:!1,isAffiliateLinksDisabled:!1,shouldBlurText:!1,shouldEnableArticleBackground:!1,variations:{}},C.displayName="ChunkedArticleContent",e.exports=C},1947:function(e,t,i){const n=i(8),o=i(0),a=i(1),r=i(82),{default:l}=i(162),{withRecircContextConsumer:s}=i(168),d=i(463),{getVariationNames:c}=i(12),p=i(39),g=i(1761),{PaymentGateway:u}=i(23),m=i(58),h=i(50),{InContent:y}=i(61),b=i(131),f=i(470),C=i(151),v=s(i(523)),w=i(1948),S=i(136),k=i(484),{showRecircMostPopular:x}=i(1950),$=i(260),{processLinks:T,processCeros:E,processTiktok:N,processSidebarHeadings:B}=i(268),{connectDomain:L}=i(18),I=L("payment"),A=L("featureFlags"),{shouldRenderNothing:P}=i(483),R=i(715),{ArticlePageChunksGrid:D}=i(1563),{ArticlePageSplitAdRail:M,ArticlePageSplitAdRailContent:O,ArticlePageSplitAdRailTop:H,ArticlePageSplitAdRailMiddle:_,ArticlePageSplitAdRailBottom:G,ArticlePageBodyGridContainer:W,LinkStackArticlePage:j}=i(1563);function F(){return o.createElement("div",null,o.createElement(h,{position:"article-mid-content",secondPosition:"in-content"}),o.createElement(y,null))}const V=new l({a:T,blockquote:({props:e})=>({type:f,props:e}),ceros:E,h2:B,tiktok:N,"cm-unit":()=>({type:F}),"inline-embed":$,"inline-recirc":e=>o.createElement(v,Object.assign({},e,{className:"article-inline-recirc-wrapper"})),"native-ad":e=>o.createElement(u,{group:"ads"},o.createElement(p,Object.assign({},e))),"inline-newsletter":e=>o.createElement(w,Object.assign({},e))});function U(e){return function(e){return Array.isArray(e)&&"string"==typeof e[0]}(e)&&e[1]&&"object"==typeof e[1]&&!Array.isArray(e[1])?e.slice(2):e.slice(1)}function z(e,t){return`${e}-${t}`}class q{constructor({adRail:e,GeneralContentWrapper:t=(()=>null),FullBleedContentWrapper:i=(()=>null),interlude:n,multiChunkRailStrategy:a,recircMostPopular:r,shouldRenderMidContent:l,shouldShowMostPopular:s,shouldUsePersistentAd:d,singleChunkRailStrategy:c,hideRecircMostPopular:p,dividerColor:g,shouldEnableArticleBackground:u,pageBackgroundTheme:m,showFirstRailRecirc:h,linkList:y,isLinkStackEnabled:b,responsiveCartoonVariation:f,hasCartoonFullWidth:v,setCartoonLinkedGalleries:w}={}){var S;this.isLinkStackEnabled=b,this.linkList=y,this.adRail=e,this.multiChunkRailStrategy=a,this.chunkCount=0,this.shouldRenderMidContent=l,this.shouldUsePersistentAd=d,this.shouldEnableArticleBackground=u,this.GeneralContentWrapper=t,this.FullBleedContentWrapper=i,this.finalAdSet=!1,this.isSingleChunk=!1,this.singleChunkRailStrategy=c,this.pageBackgroundTheme=m,this.interlude=(null===(S=null==n?void 0:n.strategy)||void 0===S?void 0:S.enabled)&&n.isRailEligible?o.createElement(C,Object.assign({},n,{isRightRail:!0})):null,this.interludeSet=!1,this.mostPopular=s&&!p&&o.createElement(k,{className:"article-recirc-most-popular-wrapper",items:r,dividerColor:g}),this.showFirstRailRecirc=h,this.responsiveCartoonVariation=f,this.hasCartoonFullWidth=v,this.setCartoonLinkedGalleries=w}determineAd(){if(this.shouldUsePersistentAd)return this.interludeSet=!0,this.persistentAd();const e=this.stickyAd(this.showFirstRailRecirc);return this.interludeSet=!0,e}showAds(e){if(1===this.chunkCount)return this.determineAd();if(!this.finalAdSet){const t=this.stickyAd(e);return this.interludeSet=!0,t}return null}getAdSlot(e){return o.createElement(o.Fragment,null,o.createElement(u,{group:"ads"},e,o.createElement(p,{position:"rail"})),o.createElement(u,{group:"consumer-marketing"},o.createElement(h,{position:"display-rail"})))}withStickyBox(e,t={}){return e&&o.createElement(b,Object.assign({},t),e)}renderSplitAdRail(){const e=this.withStickyBox(this.getAdSlot(null)),t=this.withStickyBox(this.mostPopular),i=this.withStickyBox(this.getAdSlot(null));return o.createElement(M,{anchorTop:{selector:"[data-testid='ContentHeaderLeadRailAnchor']"},anchorBottom:{selector:".content-bottom-anchor",edge:"bottom"}},o.createElement(O,{className:"split-ad-rail-content"},o.createElement(H,null,e),this.showFirstRailRecirc&&o.createElement(_,{className:"split-ad-rail-middle"},t),o.createElement(G,{className:"split-ad-rail-bottom"},i)))}renderAdRail(e){return this.isSingleChunk&&"split-in-three"===this.singleChunkRailStrategy?this.renderSplitAdRail():this.showAds(e)}closeSmallGroup(e,t,i){if(e.length>0){++this.chunkCount;const a=this.GeneralContentWrapper;return t.concat([o.createElement(D,{adRail:this.adRail,as:a,key:z("small-group",i),pageBackgroundTheme:this.pageBackgroundTheme},o.createElement(R,{className:n("body__container article__body",{"article-white-background":this.shouldEnableArticleBackground&&this.pageBackgroundTheme})},V.convert(["div",{className:"body__inner-container"},...e]),"final"===i&&this.isLinkStackEnabled&&o.createElement(j,Object.assign({},this.linkList))),this.adRail&&this.renderAdRail("final"!==i))])}return t}isMultiChunkRailStrategyAlternate(){return"alternate"===this.multiChunkRailStrategy}shouldRenderAd(){return!this.isMultiChunkRailStrategyAlternate()||this.chunkCount%2==1}shouldRenderMostPopular(e){return this.isMultiChunkRailStrategyAlternate()&&e?this.chunkCount%2==0:e}getChunkAdRailContent(e,t){return o.createElement(o.Fragment,null,(this.shouldRenderAd()||!t)&&this.getAdSlot(e),this.shouldRenderMostPopular(t)&&this.mostPopular)}injectInlineRecirc(e){let t=0,i=0;return e.reduce((n,o,a)=>this.isParagraph(o)?(t++,this.isParagraph(e[a+1])&&this.shouldInsertRecirc(t,i)?(t=0,n.concat([o,["inline-recirc",{reelId:++i}]])):n.concat([o])):n.concat([o]),[])}isParagraph(e){return e&&"p"===e[0]}persistentAd(){return o.createElement(S,{anchorTop:{selector:"[data-testid='ContentHeaderLeadRailAnchor']"},anchorBottom:{edge:"bottom"}},this.getChunkAdRailContent(this.interlude,this.showFirstRailRecirc))}shouldInsertRecirc(e,t){return e>=(0===t?5:8)}stickyAd(e){const t=!this.interludeSet&&this.interlude,i=this.getChunkAdRailContent(t,e);return o.createElement(o.Fragment,null,this.withStickyBox(i,{isExpanded:!!t}))}wrapInFullSizeContainer(e,t,i){const n=this.FullBleedContentWrapper;return t.concat([o.createElement(n,{key:z("full",i)},o.createElement(W,{className:"body__grid-container",as:R,shouldDisableMaxWidth:!0,shouldEnableDataJourneyHook:!1},V.convert(e)))])}visit(e){let t=U(e),i=[],n=[];this.isSingleChunk=!t.some(e=>"ad"===e[0]),this.finalAdSet=!1;return t.filter(this.isParagraph).length>10&&(t=this.injectInlineRecirc(t)),t.forEach((e,t)=>{var a,r,l,s;const[d,c,p]=e;if("ad"===d)this.shouldRenderMidContent&&(i=this.closeSmallGroup(n,i,t),n=[],i=i.concat([o.createElement(m,{className:"full-bleed-ad row-mid-content-ad",key:z("ad",t)},o.createElement(g,{shouldDisplayLabel:!0,shouldHoldSpace:!0}))]));else if(!this.adRail&&function(e){const t=e[0],i=e[1]||{},n="inline-embed"===t&&"callout:feature-large"===i.type,o="inline-embed"===t&&"callout:feature-medium"===i.type;return n||o||"ad"===t}(e))i=this.closeSmallGroup(n,i,t),n=[],i=this.wrapInFullSizeContainer(e,i,t);else if("inline-embed"===d&&(null===(r=null===(a=null==c?void 0:c.props)||void 0===a?void 0:a.childTypes)||void 0===r?void 0:r.includes("cartoon"))&&(null==p?void 0:p.length)){const t=p[1];t.props.image.responsiveCartoonVariation=this.responsiveCartoonVariation,t.props.image.setCartoonLinkedGalleries=this.setCartoonLinkedGalleries,n=this.hasCartoonFullWidth?n.concat([p]):n.concat([e])}else if("inline-embed"===d&&"cartoon"===(null===(s=null===(l=null==c?void 0:c.props)||void 0===l?void 0:l.image)||void 0===s?void 0:s.contentType)&&(null==e?void 0:e.length)){const t=e[1];t.props.image.responsiveCartoonVariation=this.responsiveCartoonVariation,t.props.image.setCartoonLinkedGalleries=this.setCartoonLinkedGalleries,n=n.concat([e])}else n=n.concat([e])}),i=this.closeSmallGroup(n,i,"final"),this.finalAdSet=!0,i}}function K({adRail:e,FullBleedContentWrapper:t,featureFlags:i,GeneralContentWrapper:n,interlude:a,jsonml:r,multiChunkRailStrategy:l,payment:s,recircMostPopular:d,shouldUsePersistentAd:c,singleChunkRailStrategy:p,hideRecircMostPopular:g,dividerColor:u,shouldEnableArticleBackground:m,pageBackgroundTheme:h,showFirstRailRecirc:y,wordsToDisplay:b,linkList:f,isLinkStackEnabled:C,responsiveCartoonVariation:v,hasCartoonFullWidth:w,setCartoonLinkedGalleries:S}){o.useEffect(()=>{const e=document.querySelector(".split-ad-rail-content"),t=document.querySelector(".split-ad-rail-middle"),i=document.querySelector(".split-ad-rail-bottom");if(e&&t&&i){const n=new IntersectionObserver((e=>t=>{const[i]=t;i.intersectionRatio<e&&i.target.remove()})(.95),{root:e,threshold:.95});return n.observe(t),n.observe(i),()=>{n.disconnect()}}return()=>{}});const k=!P("ads",s,i),$=x(r,b);return new q({adRail:e,FullBleedContentWrapper:t,GeneralContentWrapper:n,interlude:a,multiChunkRailStrategy:l,recircMostPopular:d,shouldRenderMidContent:k,linkList:f,isLinkStackEnabled:C,shouldShowMostPopular:$,shouldUsePersistentAd:c,singleChunkRailStrategy:p,hideRecircMostPopular:g,pageBackgroundTheme:h,dividerColor:u,shouldEnableArticleBackground:m,showFirstRailRecirc:y,responsiveCartoonVariation:v,hasCartoonFullWidth:w,setCartoonLinkedGalleries:S}).visit(r)}K.defaultProps={hasCartoonFullWidth:!1,multiChunkRailStrategy:"default",singleChunkRailStrategy:"default"},K.propTypes={adRail:a.bool,dividerColor:a.string,featureFlags:a.object,FullBleedContentWrapper:a.func,GeneralContentWrapper:a.func,hasCartoonFullWidth:a.bool,hideRecircMostPopular:a.bool,interlude:a.shape(Object.assign(Object.assign({},C.propTypes),{isRailEligible:a.boolean})),jsonml:a.array.isRequired,multiChunkRailStrategy:a.oneOf(["default","alternate"]),pageBackgroundTheme:a.string,payment:a.object,recircMostPopular:a.array,responsiveCartoonVariation:a.oneOf(c(d)),shouldUsePersistentAd:a.bool,showFirstRailRecirc:a.bool,singleChunkRailStrategy:a.oneOf(["default","split-in-three"]),wordsToDisplay:a.number},e.exports=o.memo(I(A(K)),(function(e,t){return r(e,t)})),e.exports.Chunks=K},1948:function(e,t,i){const n=i(1949),{asConfiguredComponent:o}=i(9);e.exports=o(n,"SlimNewsletterWrapper")},1949:function(e,t,i){const n=i(0),{googleAnalytics:o}=i(13),{connector:a}=i(18),r=i(469),{getNewsletterSubscriptions:l}=i(496);class s extends n.Component{constructor(e){super(e),this.fetchNewsletterSubscriptions=async e=>{try{const t=await l(e);if(200===t.status)return t.newsletterSubscriptions&&t.newsletterSubscriptions.data}catch(e){console.log(e)}return{}},this.state={}}async componentDidMount(){let e,t;const i=this.props.newsletters&&this.props.newsletters.map(e=>e.newsletterId)||[];if(document){e="nl"===new URLSearchParams(document.location.search).get("utm_source")}const{user:n,userPlatform:a}=this.props,{userPlatformProxy:r,xClientID:l}=a||{};if(n.isAuthenticated){const e={amgUUID:n.amguuid,newsletterIds:i,userPlatformProxy:r,xClientID:l,provider:"sailthru"},o=(await this.fetchNewsletterSubscriptions(e)).filter(e=>"SUBSCRIBED"===e.attributes.status).map(e=>e.attributes.newsletterId);t=i.find(e=>!o.includes(e))}else t=i[0];this.props.isNewsletterSlim&&!e&&t&&o.emitGoogleTrackingEvent("newsletter-loaded-inlineslim",{newslettterId:t}),this.setState(Object.assign(Object.assign({},this.state),{isSourceNewsletter:e,newsletterToShow:t}))}render(){let e;const{isNewsletterSlim:t,isNewsletterAggressive:i,hasSlimAlternateStyle:o,newsletterType:a,patternType:l,showToggleForLoggedInUser:s,user:d}=this.props;("aggressive-newsletter"===a&&i||"slim-newsletter"===a&&t)&&(e=!0);const{isSourceNewsletter:c,newsletterToShow:p}=this.state,g=this.props.newsletters&&this.props.newsletters.find(e=>e.newsletterId===p);return!c&&e&&g?n.createElement(r,Object.assign({},g,{enableSlimUnitToggle:s&&(null==d?void 0:d.isAuthenticated)&&(null==d?void 0:d.email),hasSlimAlternateStyle:o,newsletterType:a,patternType:l,userEmail:(null==d?void 0:d.isAuthenticated)&&(null==d?void 0:d.email)})):null}}s.propTypes=r.propTypes,s.defaultProps=r.defaultProps,s.displayName="SlimNewsletterWrapper",e.exports=a(s,{keysToPluck:["user","userPlatform"]})},1950:function(e,t,i){const{showRecircMostPopular:n}=i(1951);e.exports={showRecircMostPopular:n}},1951:function(e,t){const i=(e=[])=>Array.isArray(e)&&e.reduce((e,t,n)=>Array.isArray(t)&&t.length>1?e+i(t):"string"==typeof t&&0!==n?e+t.trim().replace(/\s+/gi," ").split(" ").length:e,0);e.exports={showRecircMostPopular:(e,t)=>i(e)>(t||200)}},1952:function(e,t,i){const n=i(22),o=i(1953),a=i(1954),r=i(1955),{connectDomain:l}=i(18),{withIncognitoDetection:s}=i(186),d=l("user"),c=l("paywall"),p=n([d,l("payment"),c,s]);e.exports={PaywallCollaborator:p(o),withArticleTruncation:a,withGalleryTruncation:r}},1953:function(e,t,i){var n=this&&this.__rest||function(e,t){var i={};for(var n in e)Object.prototype.hasOwnProperty.call(e,n)&&t.indexOf(n)<0&&(i[n]=e[n]);if(null!=e&&"function"==typeof Object.getOwnPropertySymbols){var o=0;for(n=Object.getOwnPropertySymbols(e);o<n.length;o++)t.indexOf(n[o])<0&&Object.prototype.propertyIsEnumerable.call(e,n[o])&&(i[n[o]]=e[n[o]])}return i};const o=i(1),a=i(0),r=i(516),l=i(517),{trackComponent:s}=i(5),d=e=>{a.useEffect(()=>{s("PaywallCollaborator")},[]);const{component:t,name:i,paywall:o}=e,d=n(e,["component","name","paywall"]),c=r[o.strategy],p=o.strategy&&c,g=c&&c.names.includes(i);return p&&g?a.createElement(t,Object.assign({},l.execute(c,e))):a.createElement(t,Object.assign({},d))};d.propTypes={component:o.func.isRequired,name:o.string.isRequired,payment:o.object.isRequired,paywall:o.object.isRequired},e.exports=d},1954:function(e,t,i){const n=i(1),o=i(0);e.exports=(e,t)=>{const i=e.displayName||e.name,a=e=>"object"==typeof e&&"p"===e[0],r=(e,t)=>e.filter((i,n)=>((e,t)=>e.slice(0,t).filter(a).length)(e,n)<t),l=i=>{const{[t]:n,shouldTruncate:a,paywall:{gateway:l={},paragraphLimit:s}={}}=i,d=n&&(l.shouldTruncate||a)&&(l.paragraphLimit>=0||s>=0);return o.createElement(e,Object.assign({},i,{[t]:d?r(n,l.paragraphLimit||s):n}))};return l.propTypes={[t]:n.array.isRequired,paywall:n.shape({gateway:n.shape({paragraphLimit:n.number,shouldTruncate:n.bool}),paragraphLimit:n.number}),shouldTruncate:n.bool},l.displayName=`withArticleTruncation(${i})`,l}},1955:function(e,t,i){const n=i(1),o=i(0);e.exports=(e,t)=>{const i=e.displayName||e.name,a=(e,t)=>e.map(i=>i.filter(i=>((e,t)=>e.flat().indexOf(t))(e,i)<t)).filter((e,t)=>e.length>0||0===t),r=i=>{const{[t]:n,shouldTruncate:r,paywall:{gateway:l={},gallerySlideLimit:s}={}}=i,d=n&&(l.shouldTruncate||r)&&(l.gallerySlideLimit>=0||s>=0);return o.createElement(e,Object.assign({},i,{[t]:d?a(n,l.gallerySlideLimit||s):n}))};return r.propTypes={[t]:n.array.isRequired,paywall:n.shape({gateway:n.shape({gallerySlideLimit:n.number,shouldTruncate:n.bool}),gallerySlideLimit:n.number}).isRequired,shouldTruncate:n.bool},r.displayName=`withGalleryTruncation(${i})`,r}},1956:function(e,t,i){e.exports={BeopScript:i(1957)}},1957:function(e,t,i){const n=i(0),o=i(1),{trackComponent:a}=i(5),r=({accountId:e})=>(n.useEffect(()=>{a("BeopScript")},[]),n.createElement(n.Fragment,null,n.createElement("script",{id:"beop-script",type:"text/javascript",dangerouslySetInnerHTML:{__html:`window.beOpAsyncInit = function () {\n                      window.BeOpSDK.init({\n                        account: '${e}'\n                      });\n                      window.BeOpSDK.watch();\n                  };`}}),n.createElement("script",{id:"beop-sdk",async:!0,src:"https://widget.beop.io/sdk.js"})));r.propTypes={accountId:o.string.isRequired},e.exports=r},1958:function(e,t,i){e.exports=i(1959)},1959:function(e,t,i){const n=i(1762);e.exports=n},1960:function(e,t,i){const{asConfiguredComponent:n}=i(9),o=i(1961);e.exports=n(o,"GalleryCarousel")},1961:function(e,t,i){const n=i(0),o=i(8),a=i(1),r=i(111),l=i(11),s=i(463),{getVariationNames:d}=i(12),c=i(170),{PaymentGateway:p}=i(23),g=i(1962),{EndOfPageRecirc:u}=g,{GalleryCarouselContainer:m,GalleryCarouselHeader:h,GalleryCarouselTitle:y,GalleryCarouselHeaderRecirc:b,GalleryCarouselNextWrapper:f,GalleryCarouselPrevWrapper:C,GalleryCarouselNextArrow:v,GalleryCarouselPrevArrow:w,GalleryCarouselCountWrapper:S,GalleryCarouselCount:k,GalleryCarouselContent:x,GalleryCarouselSlider:$,GalleryCarouselSliderWrapper:T}=i(1966);function E(e,t){if(!(Array.isArray(e)&&e.length>0))return[];const{midGalleryAdCadence:i,midGalleryAdFirstIndex:n,showAds:o}=t,a=o&&i,r=n||i;return e.reduce((t,n,o)=>{const l=o===e.length-1;return a&&!l&&((e,t)=>!(t!==r&&(e-r)%i!=0))(t.length,o)&&t.push({isAd:!0}),t.push(n),t},[])}const N=({items:e,midGalleryAdCadence:t,midGalleryAdFirstIndex:i,recircGalleries:a,responsiveCartoonVariation:s,shouldDisableImageClick:d,shouldHoldImageSpace:g,shouldImageLazyLoad:N,shouldUseMediumBreakpoint:B,shouldUseModalStyle:L,showAds:I,showEndRecirc:A,showHeadRecirc:P,showPublishedDate:R,sliderNavIcon:D,title:M})=>{const[O,H]=n.useState(0),[_,G]=n.useState(0),[W,j]=n.useState(!1),[F,V]=n.useState({action:new Array(2).fill(!1),slide:new Array(2).fill(null)}),U=n.useMemo(()=>E(e,{midGalleryAdCadence:t,midGalleryAdFirstIndex:i,showAds:I}),[e,t,i,I]);n.useEffect(()=>{const e={action:new Array(2).fill(!1),slide:new Array(2).fill(null)},t=_+1,i=_-1,n=U.length-1;i>=0&&(e.action[0]=!0,U[i].isAd||(e.slide[0]=i)),t<U.length&&(e.action[1]=!0,U[t].isAd||(e.slide[1]=t),t===n&&U[t].isEndRecirc&&(e.action[1]=!1)),V(e)},[_,U]);const z=e=>{e>=0&&e<=U.length&&(j(!0),H(e))};if(!(Array.isArray(U)&&U.length>0))return null;const q=_===U.length-1,K=A&&a.length>0,Z=P&&a.length>0&&a[0].url&&!(K&&q);return n.createElement(m,null,n.createElement(h,{shouldUseModalStyle:L},n.createElement(y,null,M),Z&&n.createElement(b,{href:a[0].url,target:"_blank",rel:"nofollow noopener noreferrer"},a[0].dangerousHed+" »")),n.createElement(T,null,n.createElement($,null,n.createElement(C,{isHidden:!F.action[0]},n.createElement(w,{label:"Left",isIconButton:!0,ButtonIcon:r[D],onClickHandler:()=>z(_-1),btnStyle:"text"})),n.createElement(f,{isHidden:!F.action[1]},n.createElement(v,{label:"Right",isIconButton:!0,ButtonIcon:r[D],onClickHandler:()=>z(_+1),btnStyle:"text"})),n.createElement(S,null,n.createElement(k,null,`${_+1}/${U.length}`)),U.map((e,t)=>n.createElement(x,{key:t,"data-testid":"GalleryCarouselContent_"+t,className:o({"fade-in":!W&&t===_,"fade-out":W&&t===_,"fade-in-sequence":!W&&F.slide.includes(t),"fade-out-sequence":W&&F.slide.includes(t),"current-slide":t===_,"prev-slide":t===F.slide[0],"next-slide":t===F.slide[1],"last-slide":t===U.length-1,"has-end-recric":t===U.length-1&&K}),onAnimationEnd:()=>(e=>{W&&e===_&&(G(O),j(!1))})(t),shouldDisableImageClick:d},(null==e?void 0:e.isAd)&&n.createElement(p,{group:"ads"},n.createElement(c,{position:"mid-gallery"})),["cartoon","photo"].includes((null==e?void 0:e.contentType)||"")&&n.createElement(l,Object.assign({key:e.id},e.image||e,{isLazy:N,responsiveCartoonVariation:s,shouldDisableImageClick:d,shouldHoldImageSpace:g,shouldUseMediumBreakpoint:B,showPublishedDate:R})),t===U.length-1&&K&&n.createElement(u,{items:a.slice(0,1)}))))))};N.propTypes={items:a.array.isRequired,midGalleryAdCadence:a.number,midGalleryAdFirstIndex:a.number,recircGalleries:a.array,responsiveCartoonVariation:a.oneOf(d(s)),shouldDisableImageClick:a.bool,shouldHoldImageSpace:a.bool,shouldImageLazyLoad:a.bool,shouldUseMediumBreakpoint:a.bool,shouldUseModalStyle:a.bool,showAds:a.bool,showEndRecirc:a.bool,showHeadRecirc:a.bool,showPublishedDate:a.bool,sliderNavIcon:a.string,title:a.string},N.defaultProps={midGalleryAdCadence:10,midGalleryAdFirstIndex:4,recircGalleries:[],responsiveCartoonVariation:"SliderCartoon",shouldDisableImageClick:!0,shouldHoldImageSpace:!1,shouldImageLazyLoad:!0,shouldUseMediumBreakpoint:!0,shouldUseModalStyle:!1,showAds:!1,showEndRecirc:!0,showHeadRecirc:!0,showPublishedDate:!1,sliderNavIcon:"Arrow",title:""},e.exports=N},1962:function(e,t,i){e.exports=i(1963)},1963:function(e,t,i){const{asVariation:n}=i(12),o=i(1964);o.EndOfPageRecirc=n(o,"EndOfPageRecirc",{},{isEndOfPageRecirc:!0}),e.exports=o},1964:function(e,t,i){const n=i(1),o=i(0),{useIntl:a}=i(2),r=i(49),{GalleryRecircGridWrapper:l,GalleryRecircContent:s,GalleryMidRecircHeading:d,GalleryRecircTitle:c,GalleryRecircViewGalleryCTA:p,GalleryRecircImage:g,GalleryRecircHeading:u}=i(1765),m=i(1965).default,h=({items:e,isEndOfPageRecirc:t})=>{const{formatMessage:i}=a();if(!e.length)return null;const n=i(t?m.viewNextGalleryCTAText:m.viewGalleryCTAText);return o.createElement(l,{isEndOfPageRecirc:t},!t&&o.createElement(d,null,i(m.midGalleryRecircHeading)),e.map((e,a)=>o.createElement(s,{isEndOfPageRecirc:t,"data-testid":"GalleryRecircContent",onClick:t=>((e,t)=>{e.preventDefault(),window.open(t,"_blank","noopener,noreferrer")})(t,e.url),key:e.id||a},t&&o.createElement(u,{"data-testid":"GalleryRecircHeading"},i(m.keepOnLaughingText)),o.createElement(c,{"data-testid":"GalleryRecircTitle",isEndOfPageRecirc:t},e.dangerousHed),o.createElement(p,{"data-testid":"GalleryRecircViewGalleryCTA",dangerouslySetInnerHTML:{__html:n},isEndOfPageRecirc:t}),o.createElement(g,Object.assign({isEndOfPageRecirc:t,"data-testid":"GalleryRecircImage"},e.items[0])))))};h.propTypes={isEndOfPageRecirc:n.bool,items:n.arrayOf(n.shape({dangerousHed:n.string,url:n.string,items:n.arrayOf(n.shape({altText:n.string,dangerousCaption:n.string,dangerousCredit:n.string,isDesktopPortrait:n.bool,segmentedSources:r.propTypes.segmentedSources,sources:r.propTypes.sources,links:n.arrayOf(n.shape({behavior:n.string,label:n.string.isRequired,url:n.string,network:n.string})),tagCloud:n.shape({tags:n.arrayOf(n.shape({tag:n.string.isRequired,url:n.string})),sectionHeader:n.string})}))}).isRequired).isRequired},h.defaultProps={isEndOfPageRecirc:!1},h.displayName="GalleryRecircCards",e.exports=h},1965:function(e,t,i){Object.defineProperty(t,"__esModule",{value:!0});const n=i(2);t.default=n.defineMessages({viewGalleryCTAText:{id:"GalleryRecircCards.ViewGalleryCTAText",defaultMessage:"View gallery »",description:"View gallery button text in End Of Page Recirculation."},viewNextGalleryCTAText:{id:"GalleryRecircCards.viewNextGalleryCTAText",defaultMessage:"View next gallery »",description:"View next gallery button text in End Of Page Recirculation."},keepOnLaughingText:{id:"GalleryRecircCards.keepOnLaughingText",defaultMessage:"Keep on laughing",description:"Keep on laughing text in End Of Page Recirculation."},midGalleryRecircHeading:{id:"GalleryRecircCards.midGalleryRecircHeading",defaultMessage:"Bored? Treat your attention span to another gallery.",description:"Bored? Treat your attention span to another gallery. text for mid gallery recirc"}})},1966:function(e,t,i){const{default:n,css:o,keyframes:a}=i(3),{minScreen:r,maxScreen:l,calculateSpacing:s,getColorToken:d}=i(4),{BREAKPOINTS:c}=i(6),{maxThresholds:p}=i(17),{BaseText:g,BaseWrap:u}=i(10),m=i(14),{ButtonIconWrapper:h}=i(29),{ResponsiveImageContainer:y,ResponsiveImagePicture:b}=i(24),{SpanWrapper:f}=i(85),{ResponsiveCartoonWrapper:C,ResponsiveCartoonCredit:v,ResponsiveCartoonCaption:w}=i(237),{GalleryRecircGridWrapper:S}=i(1765),k=a`
  from {
    opacity: 0;
  }

  to {
    opacity: 1;
  }
`,x=a`
  from {
    opacity: 1;
  }

  to {
    opacity: 0;
  }
`,$=a`
  from {
    opacity: 0;
  }

  to {
    opacity: .2;
  }
`,T=a`
  from {
    opacity: .2;
  }

  to {
    opacity: 0;
  }
`,E=n(u).withConfig({displayName:"GalleryCarouselContainer"})`
  display: flex;
  flex-direction: column;
  width: 100%;
  height: 100%;
`,N=n(u).withConfig({displayName:"GalleryCarouselHeader"})`
  display: flex;
  position: relative;
  align-items: center;
  justify-content: center;
  margin: 0;
  padding: ${s(2.5)} 0;
  text-align: center;

  ${l(p.lg+"px")} {
    ${({shouldUseModalStyle:e})=>e?o`
            margin-bottom: ${s(3)};
            padding: 0;
            min-height: ${s(10)};
          `:o`
            padding: ${s(5)} 0;
          `}
  }
`,B=n(g).withConfig({displayName:"GalleryCarouselTitle"})`
  ${l(p.lg+"px")} {
    width: ${s(24.75)};
  }
`;B.defaultProps={as:"h2",typeIdentity:"typography.definitions.globalEditorial.context-primary"};const L=n(g).withConfig({displayName:"GalleryCarouselHeaderRecirc"})`
  position: absolute;
  right: ${s(8)};
  text-decoration: none;

  ${l(p.lg+"px")} {
    display: none;
  }
`;L.defaultProps={as:"a",colorToken:"colors.interactive.base.brand-primary",typeIdentity:"typography.definitions.foundation.link-primary"};const I=o`
  display: flex;
  grid-row: 1;
  align-self: center;

  ${({isHidden:e})=>e&&o`
      visibility: hidden;
    `}

  ${l(p.lg+"px")} {
    display: none;
  }
`,A=n(u).withConfig({displayName:"GalleryCarouselNextWrapper"})`
  ${I};
  grid-column: 11/12;
  justify-content: start;
`,P=n(u).withConfig({displayName:"GalleryCarouselPrevWrapper"})`
  ${I};
  grid-column: 2/3;
  justify-content: end;
`,R=n(m).withConfig({displayName:"GalleryCarouselNextArrow"})`
  cursor: pointer;
  padding: 0;
  width: ${s(7.5)};

  ${h} {
    transform: translateX(0);
    transition: transform 0.3s ease;
  }

  ${h}:hover {
    transform: translateX(${s(1)});
  }
`,D=n(R).withConfig({displayName:"GalleryCarouselPrevArrow"})`
  transform: rotate(180deg);
`,M=n(u).withConfig({displayName:"GalleryCarouselCountWrapper"})`
  display: flex;
  grid-column: 10;
  grid-row: 1;
  align-items: center;
  justify-content: center;
  border: 1px solid;
  border-radius: ${s(10)};
  border-color: ${({theme:e})=>d(e,"colors.interactive.base.light")};
  width: ${s(6)};
  height: ${s(4)};

  ${l(p.lg+"px")} {
    display: none;
  }
`,O=n(g).withConfig({displayName:"GalleryCarouselCount"})`
  text-align: center;
  line-break: normal;
`;O.defaultProps={as:"p",colorToken:"colors.interactive.base.dark",typeIdentity:"typography.definitions.globalEditorial.numerical-small"};const H=n.div.withConfig({displayName:"GalleryCarouselContent"})`
  display: none;
  position: relative;
  flex-direction: row;
  align-items: start;
  justify-content: center;
  cursor: auto;
  width: 100%;
  min-height: ${s(60.25)};

  .ad--mid-gallery {
    margin: auto;
  }

  ${C} {
    border: none;
    padding: 0;

    .responsive-cartoon__caption,
    .responsive-cartoon__credit {
      overflow: hidden;
    }

    ${b} {
      cursor: zoom-in;
      text-align: center;

      ${({shouldDisableImageClick:e})=>e&&"cursor: auto;"}
    }
    ${y} {
      width: auto;
      max-width: 100%;
      height: ${s(40.5)};
    }

    ${b}, ${w}, ${v} {
      cursor: auto;
    }
  }

  ${l(p.lg+"px")} {
    display: flex;
    flex-direction: column;
    align-items: center;
    min-height: ${s(31.25)};
    ${C} {
      ${y} {
        width: ${s(40.5)};
        height: auto;
      }
    }
  }
`,_=o`
  display: flex;
  grid-row: 1;
  align-items: start;
  overflow: hidden;

  ${C} {
    ${b} {
      height: 100%;
    }

    ${y} {
      max-width: max-content;
    }
  }

  ${f} {
    position: absolute;
    min-width: ${s(40.5)};
  }
`,G=n(u).withConfig({displayName:"GalleryCarouselSlider"})`
  display: grid;
  grid-template-columns: repeat(12, 1fr);
  margin: 0 auto;
  cursor: auto;
  width: 100%;
  max-width: none;
  gap: ${s(4)};

  ${r(c.lg)} {
    .fade-in {
      animation: ${k} ease-in-out 300ms forwards;
    }

    .fade-out {
      animation: ${x} ease-in-out 300ms forwards;
    }

    .fade-in-sequence {
      animation: ${$} ease-in-out 300ms forwards;
    }

    .fade-out-sequence {
      animation: ${T} ease-in-out 300ms forwards;
    }

    .current-slide {
      display: flex;
      grid-column: 4/10;
      grid-row: 1;
    }

    .prev-slide {
      ${_};
      grid-column: 1/2;

      ${C} {
        ${b} {
          text-align: right;
        }
      }

      ${f} {
        right: 0;
      }
    }

    .next-slide {
      ${_};
      grid-column: 12/-1;

      ${C} {
        ${b} {
          text-align: left;
        }
      }

      ${f} {
        left: 0;
      }
    }

    .has-end-recric {
      ${S} {
        display: none;
      }
    }

    .has-end-recric.current-slide {
      position: unset;

      ${S} {
        display: grid;
        position: absolute;
        right: 0;
        padding-top: ${s(7)};
        width: auto;
      }
    }
  }

  ${l(p.lg+"px")} {
    display: flex;
    flex-direction: column;
    align-items: center;
    margin: 0;
    padding: 0 ${s(3)};
    gap: ${s(7)};

    ${H} {
      border-bottom: 1px solid;
      border-color: ${({theme:e})=>d(e,"colors.interactive.base.light")};
      padding-bottom: ${s(7)};

      ${C} {
        border: none;
        padding: 0;
      }
    }

    .last-slide {
      border-bottom: none;
    }

    .has-end-recric {
      gap: ${s(5)};
      padding: 0;
    }
  }
`,W=n(u).withConfig({displayName:"GalleryCarouselSliderWrapper"})`
  display: flex;
  flex-grow: 1;
  align-items: center;
  justify-content: center;
`;e.exports={GalleryCarouselContainer:E,GalleryCarouselHeader:N,GalleryCarouselTitle:B,GalleryCarouselHeaderRecirc:L,GalleryCarouselNextWrapper:A,GalleryCarouselPrevWrapper:P,GalleryCarouselNextArrow:R,GalleryCarouselPrevArrow:D,GalleryCarouselCountWrapper:M,GalleryCarouselCount:O,GalleryCarouselContent:H,GalleryCarouselSlider:G,GalleryCarouselSliderWrapper:W}},241:function(e,t,i){const{asConfiguredComponent:n}=i(9),o=i(254);e.exports=n(o,"ToggleChip")},248:function(e,t,i){const{asConfiguredComponent:n}=i(9),o=i(264);e.exports=n(o,"ToggleChipList")},254:function(e,t,i){const n=i(1),o=i(0),{useRef:a,useState:r}=i(0),{trackComponent:l}=i(5),{asThemedComponent:s}=i(30),d=i(32),{ToggleButton:c}=i(255),p=({analyticsDataAttribute:e,children:t,fullPageTheme:i,isAnchorUrl:n,isChecked:s,onChange:p,redirectUrl:g,role:u,shouldDefaultChecked:m,shouldDisplaySingleChip:h,shouldUrlRedirect:y})=>{o.useEffect(()=>{l("ToggleChip")},[]);const{current:b}=a(void 0!==s),[f,C]=r(m),v=b?s:f,[w,S]=r(v);return o.createElement(c,Object.assign({},e,{as:g?"a":"button",href:g||void 0,role:u||"switch","aria-checked":h?w:v,onClick:e=>(e=>{let i,o;if(h?S(e=>(o=!e,!e)):i=!v,b||C(i),p){const e=new CustomEvent("change",{detail:{checked:o||i}});p(e)}if(n){const e="#"+t.toString().toLowerCase();window.history.replaceState(void 0,void 0,e)}y||e.preventDefault()})(e),fullPageTheme:i}),h&&w&&o.createElement(d,null),t)};p.propTypes={analyticsDataAttribute:n.object,children:n.node.isRequired,fullPageTheme:n.oneOf(["standard","inverted"]),isAnchorUrl:n.bool,isChecked:n.bool,onChange:n.func,redirectUrl:n.string,role:n.string,shouldDefaultChecked:n.bool,shouldDisplaySingleChip:n.bool,shouldUrlRedirect:n.bool},p.defaultProps={isAnchorUrl:!1,isChecked:void 0,onChange:()=>{},shouldDefaultChecked:!1,shouldDisplaySingleChip:!1,shouldUrlRedirect:!0},e.exports=s(p)},255:function(e,t,i){const n=i(3).default,{calculateSpacing:o,getColorStyles:a,getColorToken:r,getTypographyStyles:l}=i(4),s=n.button.withConfig({displayName:"ToggleButton"})`
  display: inline-block;
  border-radius: ${o(3)};
  cursor: pointer;
  padding: ${o(1)} ${o(3)};
  text-decoration: none;
  white-space: nowrap;

  .icon {
    ${({theme:e})=>a(e,"color","colors.interactive.base.black")};
    fill: ${r("colors.interactive.base.white")};
    margin-left: ${o(-2)};
    width: ${o(4)};
    height: ${o(2)};
    vertical-align: middle;

    &:hover {
      fill: ${r("colors.interactive.base.black")};
    }
  }

  ${({theme:e})=>l(e,"typography.definitions.globalEditorial.tags")};

  &[aria-checked='false'] {
    /* TODO support rgba in getColorStyles  */
    transition: background-color 0.25s, color 0.25s;
    background-color: rgba(
      ${r("colors.interactive.base.black",{rgbOnly:!0})},
      0.1
    );
    ${({theme:e})=>a(e,"color","colors.interactive.base.black")};
  }

  &[aria-checked='true'] {
    transition: background-color 0.25s, color 0.25s;
    text-decoration: none;
    ${({theme:e})=>a(e,"background-color","colors.interactive.base.black")};
    ${({theme:e})=>a(e,"color","colors.interactive.base.white")};

    .icon {
      fill: ${r("colors.interactive.base.black")};
    }
  }

  &:hover,
  &:focus {
    outline: 0;
    /* TODO support this in getColorStyles  */
    box-shadow: 0 0 0 1px ${r("colors.interactive.base.black")}
      inset;
    text-decoration: none;

    .icon {
      fill: ${r("colors.interactive.base.black")};
    }
  }

  ${({fullPageTheme:e,theme:t})=>"inverted"===e?`\n\n      border: 1px solid;\n      ${a(t,"border-color","colors.discovery.body.white.border")};\n\n      &[aria-checked='false'] {\n        &:hover{\n          ${a(t,"background-color","colors.interactive.base.white")};\n          ${a(t,"color","colors.interactive.base.black")};\n        }\n        ${a(t,"background-color","colors.interactive.base.black")};\n        ${a(t,"color","colors.interactive.base.white")};\n      }\n\n      &[aria-checked='true'] {\n        ${a(t,"background-color","colors.interactive.base.white")};\n        ${a(t,"color","colors.interactive.base.black")};\n      }\n\n      &:hover,\n      &:focus {\n        box-shadow: none;\n      }\n    `:""}
`;e.exports={ToggleButton:s}},263:function(e,t,i){e.exports={Circle:i(271),Vogue:i(435)}},264:function(e,t,i){const n=i(1),o=i(0),{ToggleChipListWrapper:a,LabelText:r,ListWrapper:l,ListItemWrapper:s,HelperText:d}=i(130),{trackComponent:c}=i(5),{asThemedComponent:p}=i(30),g=({label:e,children:t,contentAlign:i,fullPageTheme:n,hasNoHorizontalScroll:p,isReadViewShopViewEnabled:g,hasIncreasedBottomMargin:u,hasLargeLabel:m,hasSpacing:h,hasToggleGridColor:y,helper:b,layout:f})=>(o.useEffect(()=>{c("ToggleChipList")},[]),o.createElement(a,{contentAlign:i,hasToggleGridColor:y},e&&o.createElement(r,{fullPageTheme:n,hasIncreasedBottomMargin:u,hasLargeLabel:m},e),o.createElement(l,{contentAlign:i,layout:f,isReadViewShopViewEnabled:g,hasNoHorizontalScroll:p,hasToggleGridColor:y,className:"list-wrapper"},o.Children.map(t,(e,t)=>e?o.createElement(s,{contentAlign:i,key:t,layout:f,hasSpacing:h,className:"list-item-wrapper"},e):null)),b&&o.createElement(d,null,b)));g.propTypes={children:n.array.isRequired,contentAlign:n.oneOf(["left","center","right"]),fullPageTheme:n.oneOf(["standard","inverted"]),hasIncreasedBottomMargin:n.bool,hasLargeLabel:n.bool,hasNoHorizontalScroll:n.bool,hasSpacing:n.bool,hasToggleGridColor:n.bool,helper:n.string,isReadViewShopViewEnabled:n.bool,label:n.string,layout:n.oneOf(["wrap","nowrap"])},g.defaultProps={contentAlign:"left",hasIncreasedBottomMargin:!1,hasLargeLabel:!1,hasNoHorizontalScroll:!1,hasSpacing:!1,layout:"wrap"},e.exports=p(g)},2653:function(e,t,i){var n=this&&this.__importDefault||function(e){return e&&e.__esModule?e:{default:e}};Object.defineProperty(t,"__esModule",{value:!0});const o=n(i(2654));n(i(152)).default(o.default)},2654:function(e,t,i){const n=i(2655),{asConfiguredComponent:o}=i(9);e.exports=o(n,"ArticlePage")},2655:function(e,t,i){const n=i(8),o=i(181),a=i(15),r=i(44),l=i(1),s=i(0),{injectIntl:d}=i(2),c=i(2656).default,p=i(463),{getVariationNames:g}=i(12),{I18nProvider:u}=i(531),m=i(1940),h=i(1573),y=i(2657),b=i(1945),{connector:f}=i(18),C=i(443),v=i(1481),{googleAnalytics:w}=i(13),S=i(19),{PaywallCollaborator:k,withArticleTruncation:x}=i(1952),{InlineBarrier:$}=i(61),T=i(2659),E=i(1662),N=i(151),{withLightbox:B}=i(1431),L=i(58),{Disclaimer:I}=i(97),{BeopScript:A}=i(1956),{trackComponent:P}=i(5);const R=i(2663),D=i(448),M=i(1486),O=i(469),H=i(2667),{asConfiguredComponent:_}=i(9),G=i(495),W=i(1958),j=i(444),{getNewsletterSubscriptions:F}=i(496),{ArticlePageBase:V,ArticlePageGlobalStyle:U,ArticlePageLedeBackground:z,ArticlePageContentBackGround:q,ArticlePageBodyMobileTruncatedBtn:K,ArticlePageContentFooterGrid:Z,ContentWrapperGrid:X}=i(1563),Y=i(1666),{MultiPackageRow:J}=i(1574),{getOverrideBehaviour:Q}=i(437),ee=i(2669),te=i(1960),{BaseLink:ie}=i(10);class ne extends s.Component{constructor(e){super(e),this.onHandleScroll=()=>{const e=window.scrollY<=this.pageContentEl.current.offsetTop+100;e!==this.state.hideNav&&this.setState({hideNav:e})},this.onTruncationDismiss=()=>{this.setState({isMobileTruncated:!this.state.isMobileTruncated}),w.emitUniqueGoogleTrackingEvent("article-read-more")},this.setCartoonLinkedGalleries=(e,t)=>{this.setState({sliderData:e},()=>{t()})},this.fetchNewsletterSubscriptions=async e=>{try{const t=await F(e);if(200===t.status){const e=this.props.article.newsletterModules.filter(e=>e.priority).sort((e,t)=>e.priority-t.priority).map(e=>e.newsletterId),i=this.props.article.newsletter.newsletterId,n=t.newsletterSubscriptions.data.filter(e=>"SUBSCRIBED"===e.attributes.status).map(e=>e.attributes.newsletterId);if(n.length){const t=e.filter(e=>!n.includes(e)),o=this.props.article.newsletterModules.find(e=>e.newsletterId===t[0]);n.includes(i)&&this.setState({newsletterData:Object.assign(Object.assign({},this.state.newsletterData),o)})}}}catch(e){console.log(e)}return{}},this.state={hideNav:this.props.article.headerProps.hasContentHeaderLogo,isMobileTruncated:!1,newsletterData:this.props.article.newsletter,sliderData:{}},this.pageContentEl=s.createRef();const t=e.hasLightbox?B(b,e.article.lightboxImages,e.hasSlideShow):b;this.TruncatedChunkedArticleContent=x(t,"body")}componentDidMount(){var e;if(P("ArticlePage"),this.props.user.isAuthenticated&&this.props.hasDynamicNewsletterSignup&&(null===(e=this.props.article.newsletterModules)||void 0===e?void 0:e.length)){const e=this.props.article.newsletterModules.map(e=>e.newsletterId).toString(),t={amgUUID:this.props.user.amguuid,newsletterIds:e,userPlatformProxy:this.props.userPlatform.userPlatformProxy,provider:"sailthru",xClientID:this.props.userPlatform.xClientID};this.fetchNewsletterSubscriptions(t)}const{hasTruncationOnMobile:t}=this.props.article;if(t?this.setState({isMobileTruncated:!0}):this.setState({isMobileTruncated:!1}),this.props.article.headerProps.hasContentHeaderLogo){this.setState({hideNav:!0});const e=r(this.onHandleScroll,300);window.addEventListener("scroll",e)}const i="header"===a(this.props.article.interactiveOverride,"behavior"),n=a(this.props.componentConfig,"BasePage.settings.showNavWithHeaderOverride"),o=i&&!n;window.sessionStorage.setItem("nav_invisible",o)}componentWillUnmount(){window.removeEventListener("scroll",this.onHandleScroll),window.sessionStorage.removeItem("nav_invisible")}render(){const{article:{id:e,body:t,channelCloudData:i,contentWarnings:r,affiliateDisclaimer:l,contributorSpotLightProps:d,hideAffiliateDisclaimer:p,hasAffiliateLinks:g,hasEventBannerHidden:b,hideProductDisclaimer:f,channelMap:w,coralComments:x,hasNewsletterInBody:N,hasProduct:B,hasProductLisitingCarousel:P,hasTruncationOnMobile:F,headerProps:ne,hideAutoDisclaimer:oe,hideContributorBio:ae,hideRecircList:re,hideRecircMostPopular:le,hierarchy:se,shouldReplaceOutbrainWithVMG:de,interactiveOverride:ce,isAffiliateLinksDisabled:pe,lang:ge,langTranslations:ue,customHeading:me={},recircs:he=[],recircMostPopular:ye,recircRelated:be,relatedVideo:fe,interlude:Ce,isHeroAdVisible:ve,isLicensedPartner:we,licensedPartnerLink:Se,magazineDisclaimer:ke,paddingTop:xe,tagCloud:$e,newsletter:Te,shouldUsePersistentAd:Ee,showAgeGate:Ne,showBookmark:Be,showBreadcrumbTrail:Le,showHotelRecirc:Ie,signageConfig:Ae,isLinkStackEnabled:Pe,shouldShowFooterNewsletter:Re,shouldPrioritizeSeriesPagination:De,savingsUnitedCoupons:Me=[]},showWriterBio:Oe,showFirstRailRecirc:He,showAffiliateBelowContentHeader:_e,attributes:Ge,className:We,componentConfig:je,shouldCenterDisclaimer:Fe,shouldHideBaseTopPadding:Ve,shouldHideSeriesNavigation:Ue,shouldHideSeriesRecirc:ze,pageBackgroundTheme:qe,dividerColor:Ke,shouldEnableArticleBackground:Ze,shouldInheritDropcapColor:Xe,showLinkStackInsideContentBody:Ye,featureFlags:Je,hasLightbox:Qe,hasChannelNavigation:et,showContributorSpotlight:tt,cartoonVariation:it,hasRecircDriver:nt,linkList:ot,related:at,metadataVideo:rt,outbrain:lt,taboola:st,productCarousel:dt,user:ct,hasNewsletterForABTest:pt,intl:gt,xlargePaddingTop:ut,articleVariationForXlargePaddingTop:mt,beOp:ht,hasDynamicDisclaimer:yt,responsiveCartoonVariation:bt,showGalleryCartoonPublishedDate:ft,gallerySliderTitleToken:Ct,gallerySliderTitleTarget:vt,hasGallerySliderTitleUnderline:wt}=this.props,St=Ze&&qe,kt=Ze&&Ke,{hideNav:xt}=this.state,$t=!!pt||N,{hasContentHeaderLogo:Tt}=ne;Tt&&(()=>{const e=a(je,"ContentHeader.settings");o(je,"ContentHeader.variation","TextOverlayWithLogo"),o(je,"ContentHeader.settings",Object.assign(Object.assign({},e),{showLogo:!0,hideContributors:!1,hidePublishDate:!0,hideRubric:!1,hideShareButtons:!0}))})();const Et=ae?void 0:ne.contributors,{hasNativeShareButton:Nt,hasNewsletterOnPageTop:Bt,hasNewsletterWithoutWrapper:Lt,shouldEnableNativeShareOnDesktop:It,shouldRemoveBackgroundColor:At,enableEnhancedCartoonExperience:Pt}=Je,Rt=function(e){return"WithAdRail"===a(e,"ChunkedArticleContent.variation")?S.NarrowContentWithWideAdRail:S.WithMargins}(je),Dt="OneColumnNarrow"===a(je,"ChunkedArticleContent.variation"),Mt=S.DynamicGrid({startColumn:2,endColumn:12}),Ot=Fe?I.TextCenterNoTopRule:I,Ht=he.map((e,t)=>{const i=_(G,e.displayName),n=e.heading||"",{results:o,destinationPage:a,location:r,recommendedClickout:l,recommendedType:d}=e;return nt&&a?s.createElement(W,{key:"SummaryCollectionSplitColumns"+t,recommendedItems:{items:o,recommendedType:d,recommendedClickout:l},guideItem:[a],location:r,shouldAppendReadMoreLinkForDek:!0}):s.createElement(j,{key:"ConnectedErrorBoundary"+t},s.createElement(i,{related:e.related,heading:n,dividerColor:kt}))}),_t=a(je,"ContentHeader.variation")===mt&&ut?ut:xe,Gt=(null==ht?void 0:ht.accountID)||"",Wt=(null==ht?void 0:ht.isEnabled)||!1,jt=et&&s.createElement(D,null),Ft="WithAdRail"===a(je,"ChunkedArticleContent.variation"),Vt=({children:e})=>s.createElement(X,{isadRail:Ft,as:Rt},s.createElement("div",{className:"body body__container"},s.createElement("div",{className:"container container--body"},s.createElement("div",{className:"container--body-inner"},e)))),Ut="articleheader"===Q(ce),zt=a(ne,"sponsoredContentHeaderProps"),qt=s.createElement(s.Fragment,null,s.createElement(L,null,s.createElement(Mt,null,s.createElement(Ot,{disclaimerHtml:l,hasTopRule:!1}))));return s.createElement(V,{additionalNavigation:jt,attributes:Ge,channelMap:w,className:n("page--article",We),featureFlags:Je,hideNav:xt,hasContentHeaderLogo:Tt,hasEventBannerHidden:b,hasFooterAdsMargins:!0,interactiveOverride:ce,isHeroAdVisible:ve,hasBaseAds:!0,user:ct,lang:ge,customHeading:me,shouldHideBaseTopPadding:Ve,shouldPrioritizeSeriesPagination:De,pageBackgroundTheme:St},Wt&&s.createElement(A,{accountId:Gt}),s.createElement(u,{locale:ge,translations:ue},Be&&s.createElement(M,null),g&&l&&!p&&!f&&!_e&&qt,Le&&s.createElement(h,{hierarchy:se,shouldRemoveBackgroundColor:At}),s.createElement("article",{className:n("article main-content",{"article--inherited-dropcaps":Xe}),lang:ge},Bt&&Te&&s.createElement(O,Object.assign({},Te,{position:"article-page-top"})),ne.sponsoredContentHeaderProps&&s.createElement(E,Object.assign({},ne.sponsoredContentHeaderProps,{className:"light-theme"})),Ut?s.createElement("div",{className:"interactive-override-container interactive-override-container--content_header",dangerouslySetInnerHTML:{__html:ce.markup}}):s.createElement(z,{ref:this.pageContentEl},s.createElement(v,Object.assign({},ne,{hasNativeShareButton:Nt,shouldEnableNativeShareOnDesktop:It,className:"article__content-header",hasLightbox:Qe,stickySocialAnchorBottom:{selector:".content-bottom-anchor",edge:"bottom"},stickySocialAnchorTop:{selector:".body",edge:"top"},interactiveOverride:ce,metadataVideo:rt,showBreadCrumb:Le}))),!Ue&&s.createElement(R,{className:"article__series-navigation",pageBackgroundTheme:St,dividerColor:kt}),s.createElement(q,{togglePaddingTop:_t,isMobileTruncated:this.state.isMobileTruncated,cartoonVariation:it,"data-attribute-verso-pattern":"article-body"},F&&this.state.isMobileTruncated&&s.createElement(K,{inputKind:"button",label:gt.formatMessage(c.truncatedButtonLabel),onClickHandler:this.onTruncationDismiss}),a(je,"ChannelCloud.settings.shouldShowChannelCloud",!1)&&(null===(Kt=null==i?void 0:i.channels)||void 0===Kt?void 0:Kt.length)>0&&s.createElement(S.ContentWithAdRailNarrow,null,s.createElement("div",null,s.createElement(H,Object.assign({},i,a(je,"ChannelCloud.settings"))))),g&&l&&!p&&!f&&_e&&qt,t&&s.createElement(k,{body:t,linkList:ot,isLinkStackEnabled:Pe&&Ye,component:this.TruncatedChunkedArticleContent,contributors:Et,hasProduct:B,hasProductLisitingCarousel:P,hasTopSpacing:!!ne.lede,interlude:Ce,isAffiliateLinksDisabled:pe,name:"chunked-article-content",shouldUsePersistentAd:Ee,hideRecircMostPopular:le,shouldEnableArticleBackground:Ze,pageBackgroundTheme:St,dividerColor:kt,recircMostPopular:ye,showFirstRailRecirc:He,hideAutoDisclaimer:oe,hideAffiliateDisclaimer:p,hideProductDisclaimer:f,responsiveCartoonVariation:bt,hasCartoonFullWidth:Pt,setCartoonLinkedGalleries:this.setCartoonLinkedGalleries}),t&&s.createElement(Vt,null,s.createElement($,null))),!ze&&s.createElement(T,{ContentWrapper:Rt})),Wt&&s.createElement(Vt,null,s.createElement("div",{className:"BeOpWidget"})),Object.keys(dt).length>0&&s.createElement(J,{key:"articleCarouselProduct",dataJourneyHook:"row-content"},s.createElement(Y,Object.assign({},dt,{copilotId:e}))),s.createElement(Z,{as:C,className:n({"content-footer--mobile-truncated":this.state.isMobileTruncated}),channelMap:w,consumerMarketing:{position:"article-footer"},ContentWrapper:Rt,contributors:Et,contributorSpotlight:d,coralComments:x,showWriterBio:Oe,hideContributorBio:ae,showContributorSpotlight:tt,hideRecircList:re,hasNewsletterWithoutWrapper:Lt,isLicensedPartner:we,isLinkStackEnabled:Pe&&!Ye,isNarrow:Dt,isAdRail:Ft,licensedPartnerLink:Se,linkList:ot,magazineDisclaimer:ke,newsletter:ct.isAuthenticated?this.state.newsletterData:Te,outbrain:lt,taboola:st,recircs:he,recircListElements:Ht,dividerColor:kt,related:at,recircRelated:be,relatedVideo:fe,showNewsletter:Re||!$t,showHotelRecirc:Ie,signageConfig:Ae,tagCloud:$e,shouldReplaceOutbrainWithVMG:de,savingsUnitedCoupons:Me,sponsoredProps:zt,hasDynamicDisclaimer:yt}),Ne&&s.createElement(m,{content:{contentWarnings:r}}),!ct.isAuthenticated&&ct.hasUserAuthCheck&&s.createElement(y,{as:"document",hint:"prefetch",href:"/account/sign-in"})),Pt&&s.createElement(ee,{closeModalText:gt.formatMessage(c.backToArticle)},s.createElement(te,{items:this.state.sliderData.items,showPublishedDate:ft,sliderNavIcon:"WavyArrow",recircGalleries:this.state.sliderData.recricGalleries,responsiveCartoonVariation:"SliderCartoon",title:this.state.sliderData.dangerousHed&&s.createElement(ie,{href:""+this.state.sliderData.url,target:vt,hasUnderline:wt,typeToken:Ct},this.state.sliderData.dangerousHed),shouldUseModalStyle:!0})),s.createElement(U,{pageBackgroundTheme:St,dividerColor:kt}));var Kt}}ne.propTypes={article:l.shape({body:l.array,channelCloudData:l.object,affiliateDisclaimer:l.string,contributorSpotLightProps:l.object,hideProductDisclaimer:l.bool,hideAffiliateDisclaimer:l.boolean,hasAffiliateLinks:l.boolean,channelMap:l.object,coralComments:l.shape({enableComments:l.bool,coralHostName:l.string}),contentWarnings:l.array,customHeading:l.object,hasEventBannerHidden:l.bool,hasNewsletterInBody:l.bool,hasProduct:l.bool,hasProductLisitingCarousel:l.bool,hasTruncationOnMobile:l.bool,headerProps:l.object.isRequired,hideAutoDisclaimer:l.bool,hideContributorBio:l.bool,hideRecircList:l.bool,hideRecircMostPopular:l.bool,hierarchy:l.array,id:l.string,interactiveOverride:l.shape({markup:l.string,behavior:l.string}),interlude:l.shape(Object.assign(Object.assign({},N.propTypes),{isRailEligible:l.boolean})),isAffiliateLinksDisabled:l.bool,isHeroAdVisible:l.bool.isRequired,isLicensedPartner:l.bool,isLinkStackEnabled:l.bool,lang:l.string,langTranslations:l.object,licensedPartnerLink:l.string,lightboxImages:l.array.isRequired,magazineDisclaimer:l.shape({issueDate:l.string.isRequired,issueLink:l.string.isRequired,originalHed:l.string}),newsletter:l.object,newsletterModules:l.array,paddingTop:l.oneOf(["large"]),recircs:l.array,recircMostPopular:l.array,recircRelated:l.array,relatedVideo:l.shape({brand:l.string,related:l.any,useRelatedVideo:l.bool}),savingsUnitedCoupons:l.array,shouldPrioritizeSeriesPagination:l.bool,shouldShowFooterNewsletter:l.bool,shouldUsePersistentAd:l.bool,shouldReplaceOutbrainWithVMG:l.bool,showAgeGate:l.bool,showAffiliateBelowContentHeader:l.bool,showBookmark:l.bool,showBreadcrumbTrail:l.bool,showHotelRecirc:l.bool,signageConfig:l.object,tagCloud:l.shape({tags:l.arrayOf(l.shape({tag:l.string.isRequired,url:l.string}))})}).isRequired,articleVariationForXlargePaddingTop:l.string,attributes:l.object,beOp:l.shape({accountID:l.string,isEnabled:l.boolean}),cartoonVariation:l.oneOf(["default","card"]),className:l.string,componentConfig:l.object,dividerColor:l.string,featureFlags:l.object,gallerySliderTitleTarget:l.string,gallerySliderTitleToken:l.string,hasChannelNavigation:l.bool,hasDynamicDisclaimer:l.bool,hasDynamicNewsletterSignup:l.bool,hasGallerySliderTitleUnderline:l.bool,hasLightbox:l.bool,hasNewsletterForABTest:l.bool,hasRecircDriver:l.bool,hasSlideShow:l.bool,hideNav:l.bool,intl:l.object,linkList:l.object,metadataVideo:l.shape({isLive:l.bool,premiereDate:l.string,series:l.string,videoLength:l.number,premiereGap:l.number}),outbrain:l.shape({canonicalUrl:l.string.isRequired,shouldDisplayAboveRecircList:l.bool,template:l.string.isRequired,widgetId:l.string}),pageBackgroundTheme:l.string,productCarousel:l.object,related:l.array,responsiveCartoonVariation:l.oneOf(g(p)),shouldCenterDisclaimer:l.bool,shouldEnableArticleBackground:l.bool,shouldHideBaseTopPadding:l.bool,shouldHideSeriesNavigation:l.bool,shouldHideSeriesRecirc:l.bool,shouldInheritDropcapColor:l.bool,showAffiliateBelowContentHeader:l.bool,showContributorSpotlight:l.bool,showFirstRailRecirc:l.bool,showGalleryCartoonPublishedDate:l.bool,showLinkStackInsideContentBody:l.bool,showWriterBio:l.bool,taboola:l.shape({publisherId:l.string.isRequired}),user:l.object,userPlatform:l.object,xlargePaddingTop:l.string},ne.defaultProps={cartoonVariation:"default",gallerySliderTitleTarget:"_blank",gallerySliderTitleToken:"typography.definitions.globalEditorial.context-primary",hasDynamicNewsletterSignup:!1,hasGallerySliderTitleUnderline:!1,hasLightbox:!1,hasSlideShow:!0,metadataVideo:{},productCarousel:{},related:[],shouldCenterDisclaimer:!1,shouldEnableArticleBackground:!1,shouldHideSeriesNavigation:!0,shouldHideSeriesRecirc:!0,shouldInheritDropcapColor:!1,showAffiliateBelowContentHeader:!1,showGalleryCartoonPublishedDate:!0,showLinkStackInsideContentBody:!1,showWriterBio:!1},ne.displayName="ArticlePage",e.exports=f(d(ne),{keysToPluck:["article","beOp","componentConfig","featureFlags","linkList","metadataVideo","outbrain","productCarousel","related","showFirstRailRecirc","taboola","user","userPlatform"]})},2656:function(e,t,i){Object.defineProperty(t,"__esModule",{value:!0});const n=i(2);t.default=n.defineMessages({truncatedButtonLabel:{id:"ArticlePage.TruncatedButtonLabel",defaultMessage:"Read Full Story",description:"ArticlePage component truncated button label"},backToArticle:{id:"ArticlePage.Back to article",defaultMessage:"Back to article",description:"Gallery slider back button text"}})},2657:function(e,t,i){e.exports=i(2658)},2658:function(e,t,i){const n=i(0),{trackComponent:o}=i(5);class a extends n.PureComponent{constructor(){super(...arguments),this.writeResourceHintLink=e=>{const{as:t,hint:i,href:n}=e,o=document.createElement("link");o.as=t,o.href=n,o.rel=i,document.head.appendChild(o)}}componentDidMount(){o("ResourceHint"),this.props&&this.writeResourceHintLink(this.props)}render(){return n.createElement(n.Fragment,null)}}e.exports=a},2659:function(e,t,i){const{asConfiguredComponent:n}=i(9),o=i(2660);e.exports=n(o,"SeriesRecirc")},2660:function(e,t,i){const n=i(1),o=i(0),{useIntl:a}=i(2),{useInView:r}=i(60),{connector:l}=i(18),s=i(11),d=i(19),c=i(2661).default,{trackComponent:p}=i(5),{googleAnalytics:g}=i(13),{SeriesRecircAsset:u,SeriesRecircContainer:m,SeriesRecircContent:h,SeriesRecircDek:y,SeriesPromoHed:b,SeriesRecircReadMoreCta:f,SeriesRecircReadMoreCtaMobile:C,SeriesRecircTextContainer:v}=i(2662),w={"Read more":c.readMoreDefault,"Read next":c.readNext},S=({ContentWrapper:e,readMoreCTA:t,seriesData:i})=>{const{formatMessage:n}=a();o.useEffect(()=>{p("SeriesRecirc")},[]);const l=(({links:e})=>{let t;for(let i=0;i<e.length;i++){const{isCurrent:n}=e[i];if(n){for(let n=i+1;n<e.length;n++){const{isExternal:i,isPublished:o}=e[n]||{};if(o&&!i){t=e[n];break}}break}}return t})({links:(null==i?void 0:i.links)||[]}),d=e=>{g.emitGoogleTrackingEvent("seriesrecirc",{title:e})},[c,S]=r();if(o.useEffect(()=>{S&&g.emitUniqueGoogleTrackingEvent("series-inview",{title:null==l?void 0:l.promoHed})},[S]),!i)return null;if(!l)return null;const{dek:k="",hed:x="",image:$=null,promoHed:T="",url:E=""}=l,N=$&&($.segmentedSources||$.sources)&&Object.assign({},$);return o.createElement(e,null,l&&o.createElement(m,{ref:c},o.createElement(h,null,o.createElement(C,null,n(w[t])),o.createElement(u,null,N&&o.createElement("a",{href:E,onClick:()=>d(T||x)},o.createElement(s,Object.assign({},N,{isLazy:!0})))),o.createElement(v,null,o.createElement(f,null,n(w[t])),(x||T)&&o.createElement(b,null,o.createElement("a",{href:E,onClick:()=>d(T||x)},o.createElement("span",{dangerouslySetInnerHTML:{__html:T||x}}))),k&&o.createElement(y,null,o.createElement("span",{dangerouslySetInnerHTML:{__html:k}}))))))};S.propTypes={ContentWrapper:n.elementType,readMoreCTA:n.string,seriesData:n.shape({hed:n.string,dek:n.string,image:n.object,links:n.arrayOf(n.shape({hed:n.string,dek:n.string,image:n.object,isCurrent:n.bool,isExternal:n.bool,isPublished:n.bool,promoDek:n.string,promoHed:n.string,url:n.string}))})},S.defaultProps={ContentWrapper:d.NarrowContentWithWideAdRail,readMoreCTA:"Read more",seriesData:null},S.displayName="SeriesRecirc",e.exports=l(S,{keysToPluck:["seriesData"]})},2661:function(e,t,i){Object.defineProperty(t,"__esModule",{value:!0});const n=i(2);t.default=n.defineMessages({readMoreDefault:{id:"ReadMore.SeriesRecirc",defaultMessage:"Read more",description:"SeriesRecirc component Read more text"},readNext:{id:"ReadNext.SeriesRecirc",defaultMessage:"Read next",description:"SeriesRecirc component Read next text"}})},2662:function(e,t,i){const n=i(3).default,{BaseText:o}=i(10),{calculateSpacing:a,getColorToken:r}=i(4),{BREAKPOINTS:l}=i(6),s=n.div.withConfig({displayName:"SeriesRecircAsset"})`
  grid-column: 1/5;

  @media (max-width: ${l.md}) {
    grid-column: 1/-1;
  }
`,d=n.div.withConfig({displayName:"SeriesRecircContainer"})`
  margin-top: ${a(4)};
  margin-bottom: ${a(5)};
  border-top: 2px solid
    ${({theme:e})=>r(e,"colors.discovery.body.white.heading")};
  padding-top: ${a(2)};
`,c=n.figure.withConfig({displayName:"SeriesRecircContent"})`
  margin-right: 0;
  margin-left: 0;

  @media (min-width: ${l.md}) {
    display: grid;
    grid-column-gap: ${a(3)};
    grid-template-columns: repeat(10, 1fr);
  }

  @media (max-width: ${l.md}) {
    display: block;
  }
`,p=n(o).withConfig({displayName:"SeriesRecircDek"})`
  @media (min-width: ${l.md}) {
    grid-column: 1/-1;
  }
`;p.defaultProps={as:"div",typeIdentity:"typography.definitions.consumptionEditorial.description-embed"},p.displayName="SeriesRecircDek";const g=n(o).withConfig({displayName:"SeriesPromoHed"})`
  a {
    text-decoration: none;
    color: inherit;

    &:hover {
      text-decoration: underline;
    }
  }

  @media (min-width: ${l.md}) {
    grid-column: 1/-1;
  }
`;g.defaultProps={as:"h2",bottomSpacing:.625,topSpacing:1,typeIdentity:"typography.definitions.discovery.subhed-section-tertiary"};const u=n(o).withConfig({displayName:"SeriesRecircReadMoreCta"})`
  @media (max-width: ${l.md}) {
    display: none;
  }
`;u.defaultProps={typeIdentity:"typography.definitions.discovery.subhed-section-primary"};const m=n(u).withConfig({displayName:"SeriesRecircReadMoreCtaMobile"})`
  display: none;

  @media (max-width: ${l.md}) {
    display: block;
    grid-column: 1/-1;
    margin-bottom: 1em;
  }
`,h=n.div.withConfig({displayName:"SeriesRecircTextContainer"})`
  @media (min-width: ${l.md}) {
    grid-column: 5/-1;
  }
`;e.exports={SeriesRecircAsset:s,SeriesRecircContainer:d,SeriesRecircContent:c,SeriesRecircDek:p,SeriesPromoHed:g,SeriesRecircReadMoreCta:u,SeriesRecircReadMoreCtaMobile:m,SeriesRecircTextContainer:h}},2663:function(e,t,i){const{asConfiguredComponent:n}=i(9),o=i(2664);e.exports=n(o,"SeriesNavigation")},2664:function(e,t,i){const n=i(1),o=i(0),a=i(8),{useIntl:r}=i(2),{connector:l}=i(18),s=i(11),d=i(2665).default,{googleAnalytics:c}=i(13),{trackComponent:p}=i(5),{SeriesNavigationAsset:g,SeriesNavigationItemContainer:u,SeriesNavigationDek:m,SeriesNavigationHeadingContainer:h,SeriesNavigationHed:y,SeriesNavigationItemDek:b,SeriesNavigationItemHed:f,SeriesNavigationItemNowReading:C,SeriesNavigationList:v,SeriesNavigationListItem:w,SeriesNavigationResponsiveAssetComingSoonText:S,SeriesNavigationResponsiveAssetContainer:k,SeriesNavigationTextContainer:x,SeriesNavigationWrapper:$,UnpublishedResponsiveAssetContainer:T}=i(2666),E={"Coming soon":d.comingSoonText,"Now reading":d.nowReadingText},N=({comingSoonText:e,nowReadingText:t,links:i})=>{const{formatMessage:n}=r(),l=a("grid"),d=e=>{c.emitGoogleTrackingEvent("seriesnavigation",{title:e})};return i.length?o.createElement(v,{className:l},i.map((i={},a)=>{const r=a,{dek:l,hed:c,image:p,isCurrent:m,isExternal:h,isPublished:y,url:v}=i,$=!y&&!h,N=p&&(p.segmentedSources||p.sources);return o.createElement(w,{key:r},o.createElement(u,null,N&&o.createElement(g,{isComingSoon:$},o.createElement(k,{isComingSoon:$},$?o.createElement(o.Fragment,null,o.createElement(S,null,n(E[e])),o.createElement(T,null,o.createElement(s,Object.assign({},p,{isLazy:!0})))):o.createElement("a",{href:v,onClick:()=>d(c)},o.createElement(s,Object.assign({},p,{isLazy:!0}))))),o.createElement(x,null,c&&($?o.createElement(f,{isComingSoon:$,dangerouslySetInnerHTML:{__html:c}}):o.createElement(f,null,o.createElement("a",{href:v,dangerouslySetInnerHTML:{__html:c},onClick:()=>d(c)}))),!m&&l&&o.createElement(b,{isComingSoon:$,dangerouslySetInnerHTML:{__html:l}}),m&&o.createElement(C,{isCurrent:m,dangerouslySetInnerHTML:{__html:n(E[t])}}))))})):null};N.propTypes={comingSoonText:n.string,links:n.arrayOf(n.shape({dek:n.string,hed:n.string,isExternal:n.bool,url:n.string})),nowReadingText:n.string},N.defaultProps={links:[]};const B=({className:e,comingSoonText:t,nowReadingText:i,dividerColor:n,pageBackgroundTheme:r,seriesData:l})=>{if(o.useEffect(()=>{p("SeriesNavigation")},[]),!l)return null;const{hed:s,dek:d,links:c}=l,g={comingSoonText:t,hed:s,dek:d,links:c,nowReadingText:i},u=a("grid",e);return o.createElement($,{className:u,pageBackgroundTheme:r,dividerColor:n},o.createElement(h,null,s&&o.createElement(y,{dangerouslySetInnerHTML:{__html:s}}),d&&o.createElement(m,{dangerouslySetInnerHTML:{__html:d}})),o.createElement(N,Object.assign({},g)))};B.displayName="SeriesNavigation",B.propTypes={className:n.string,comingSoonText:n.string,dividerColor:n.string,nowReadingText:n.string,pageBackgroundTheme:n.string,seriesData:n.shape({hed:n.string,dek:n.string,image:n.object,links:n.arrayOf(n.shape({hed:n.string,dek:n.string,image:n.object,isCurrent:n.bool,isExternal:n.bool,isPublished:n.bool,promoHed:n.string,url:n.string}))})},B.defaultProps={comingSoonText:"Coming soon",nowReadingText:"Now reading",seriesData:null},e.exports=l(B,{keysToPluck:["seriesData"]})},2665:function(e,t,i){Object.defineProperty(t,"__esModule",{value:!0});const n=i(2);t.default=n.defineMessages({comingSoonText:{id:"ComingSoon.SeriesNavigation",defaultMessage:"COMING SOON",description:"ChannelFilter component Coming Soon text"},nowReadingText:{id:"NowReading.SeriesNavigation",defaultMessage:"Now Reading",description:"SeriesNavigation component Now Reading text"}})},2666:function(e,t,i){const n=i(3).default,{applyGridSpacing:o,cssVariablesGrid:a,applyCustomDividerColor:r}=i(16),{BaseText:l}=i(10),{calculateSpacing:s,getTypographyStyles:d,getColorStyles:c}=i(4),{BREAKPOINTS:p}=i(6),{universalGridCore:g}=i(57),{applyCustomBackgroundColor:u}=i(16),m=n.div.withConfig({displayName:"SeriesNavigationAsset"})`
  ${({isComingSoon:e})=>e?"background: black;":""}
`,h=n.div.withConfig({displayName:"SeriesNavigationItemContainer"})`
  @media (max-width: ${p.md}) {
    width: ${s(22)};
  }
`,y=n.div.withConfig({displayName:"SeriesNavigationDek"})`
  ${({theme:e})=>d(e,"typography.definitions.discovery.description-page")};
`,b=n.div.withConfig({displayName:"SeriesNavigationHeadingContainer"})`
  grid-column: 1/-1;
  padding-top: ${s(2)};
`,f=n.h2.withConfig({displayName:"SeriesNavigationHed"})`
  ${({theme:e})=>d(e,"typography.definitions.discovery.hed-core-primary")};

  margin-top: 0;
  margin-bottom: ${s(.25)};
`,C=n(l).withConfig({displayName:"SeriesNavigationItemDek"})`
  ${({theme:e})=>d(e,"typography.definitions.globalEditorial.context-secondary")};
  ${({isComingSoon:e})=>e?"opacity: 43%":""}
`;C.defaultProps={isComingSoon:!1};const v=n(l).withConfig({displayName:"SeriesNavigationItemHed"})`
  ${({isComingSoon:e})=>e?"opacity: 43%;":""};

  a {
    text-decoration: none;
    color: inherit;

    &:hover {
      text-decoration: underline;
    }
  }
`;v.defaultProps={as:"p",bottomSpacing:.6875,isComingSoon:!1,topSpacing:1,typeIdentity:"typography.definitions.discovery.hed-bulletin-primary"};const w=n(l).withConfig({displayName:"SeriesNavigationItemNowReading"})`
  ${({theme:e})=>d(e,"typography.definitions.globalEditorial.context-secondary")};

  ${({theme:e})=>c(e,"color","colors.interactive.feedback.invalid-primary")};
`,S=n.ul.withConfig({displayName:"SeriesNavigationList"})`
  grid-column: 1/-1;
  margin: 0;
  padding: 0 0 0.5rem 0;
  height: 100%;
  list-style: none;

  &.grid {
    ${g()}
  }
  @media (max-width: ${p.md}) {
    &.grid {
      display: flex;
      flex-wrap: nowrap;
      align-items: flex-start;
      padding: 0 0 1rem 0;
      width: 100%;
      overflow-x: scroll;
    }
  }
`,k=n.li.withConfig({displayName:"SeriesNavigationListItem"})`
  display: inline-block;
  grid-column: span 3;

  @media (max-width: ${p.md}) {
    &:last-child {
      padding-right: ${s(3)};
    }
  }

  @media (min-width: ${p.lg}) {
    grid-column: span 2;
  }
`,x=n.div.withConfig({displayName:"SeriesNavigationResponsiveAssetContainer"})`
  position: relative;
`,$=n(l).withConfig({displayName:"SeriesNavigationResponsiveAssetComingSoonText"})`
  position: absolute;
  top: ${s(1)};
  left: ${s(1.25)};
  z-index: 1;
`;$.defaultProps={as:"span",colorToken:"colors.background.adContainer.special",typeIdentity:"typography.definitions.discovery.subhed-section-secondary"};const T=n.div.withConfig({displayName:"SeriesNavigationWrapper"})`
  ${a()}
  ${o("padding")};

  grid-template-rows: auto auto;
  border-bottom: 1px solid;
  padding-bottom: ${s(1)};
  width: 100%;

  ${({dividerColor:e,theme:t})=>""+(e?r(e)+";":c(t,"border-color","colors.consumption.body.standard.divider")+";")}

  ${({pageBackgroundTheme:e,theme:t})=>""+(e?u(e)+";":c(t,"background","colors.foundation.menu-bg.collapsed")+";")}
  &.grid {
    ${g()}
  }
`,E=n.div.withConfig({displayName:"SeriesNavigationTextContainer"})`
  margin-top: ${s(2)};
`,N=n.div.withConfig({displayName:"UnpublishedResponsiveAssetContainer"})`
  opacity: 43%;
`;e.exports={SeriesNavigationAsset:m,SeriesNavigationItemContainer:h,SeriesNavigationDek:y,SeriesNavigationHeadingContainer:b,SeriesNavigationHed:f,SeriesNavigationItemDek:C,SeriesNavigationItemHed:v,SeriesNavigationItemNowReading:w,SeriesNavigationList:S,SeriesNavigationListItem:k,SeriesNavigationResponsiveAssetComingSoonText:$,SeriesNavigationResponsiveAssetContainer:x,SeriesNavigationTextContainer:E,SeriesNavigationWrapper:T,UnpublishedResponsiveAssetContainer:N}},2667:function(e,t,i){const n=i(1),o=i(0),{asConfiguredComponent:a}=i(9),{googleAnalytics:r}=i(13),{trackComponent:l}=i(5),{ChannelCloudContainer:s,ChannelCloudContainerWrapper:d,ChannelCloudHeaderContainer:c,ChannelCloudHeaderLink:p,ChannelCloudHeaderImage:g,ChannelCloudSubChannelContainer:u,ChannelCloudSubChannelLink:m,ChannelCloudSubChannelLinkText:h,ChannelCloudSubChannelText:y}=i(2668),b=({channels:e,headerLogo:t,headerLink:i,sectionHeader:n})=>(o.useEffect(()=>{l("ChannelCloud")},[]),o.createElement(d,null,n&&o.createElement(c,null,t&&o.createElement(g,{src:t,alt:"logo"}),o.createElement(p,{href:i,hasLogo:Boolean(t),dangerouslySetInnerHTML:{__html:n}})),e&&o.createElement(s,null,e.map(e=>o.createElement(u,{key:e.id},o.createElement(y,{dangerouslySetInnerHTML:{__html:e.text}}),e.sub.map(e=>o.createElement(m,{key:e.id,href:e.url,onClick:()=>r.emitGoogleTrackingEvent("channelCloud",e)},o.createElement(h,{dangerouslySetInnerHTML:{__html:e.text}}))))))));b.propTypes={channels:n.arrayOf(n.shape({id:n.string,text:n.string,originalText:n.string,sub:n.arrayOf(n.shape({id:n.string,text:n.string,url:n.string}))})).isRequired,headerLink:n.string,headerLogo:n.string,sectionHeader:n.string.isRequired},b.displayName="ChannelCloud",e.exports=a(b,"ChannelCloud")},2668:function(e,t,i){const n=i(3).default,{BaseText:o,BaseLink:a}=i(10),{calculateSpacing:r,getColorStyles:l}=i(4),s="\n  display: flex;\n  align-items: center;\n",d=n.div.withConfig({displayName:"ChannelCloudContainerWrapper"})``,c=n.div.withConfig({displayName:"ChannelCloudHeaderContainer"})`
  ${s};

  border-width: 0 0 ${r(.25)};
  border-style: solid;
  padding: ${r(1)} ${r(3)} ${r(2)}
    0;

  ${({theme:e})=>l(e,"border-color","colors.interactive.base.black")};
`,p=n.img.withConfig({displayName:"ChannelCloudHeaderImage"})`
  width: 25px;
  height: 30px;
`,g=n(a).withConfig({displayName:"ChannelCloudHeaderLink"})`
  position: relative;
  top: ${r(.3)};
  padding-left: ${({hasLogo:e})=>e?r(1.3):0};
`;g.defaultProps={colorToken:"colors.interactive.base.black",typeIdentity:"typography.definitions.discovery.hed-bulletin-secondary"};const u=n.div.withConfig({displayName:"ChannelCloudContainer"})`
  display: flex;
  flex-wrap: wrap;
  padding: ${r(2.4)} ${r(6)}
    ${r(1)} 0;
`,m=n.div.withConfig({displayName:"ChannelCloudSubChannelContainer"})`
  ${s}
  flex-wrap: wrap;
  margin-bottom: ${r(2)};
  padding-right: ${r(2)};
`,h=n(o).withConfig({displayName:"ChannelCloudSubChannelText"})`
  padding-right: ${r(1)};

  &::after {
    content: ':';
  }
`;h.defaultProps={colorToken:"colors.interactive.base.black",typeIdentity:"typography.definitions.globalEditorial.context-primary"};const y=n(a).withConfig({displayName:"ChannelCloudSubChannelLink"})`
  ${s}
  padding-right: ${r(1)};

  svg {
    ${({theme:e})=>l(e,"fill","colors.consumption.body.standard.body-deemphasized")};

    position: relative;
    top: 2px;
    right: 2px;
    transform: rotate(-45deg);
    width: 12px;
    height: 12px;
    vertical-align: bottom;
  }

  &::after {
    ${({theme:e})=>l(e,"color","colors.consumption.body.standard.body-deemphasized")};

    position: relative;
    right: ${({hasIcon:e})=>e?r(.4):0};
    line-height: 0;
    content: ',';
  }

  &:last-child {
    &::after {
      content: '';
    }
  }
`,b=n(o).withConfig({displayName:"ChannelCloudSubChannelLinkText"})`
  line-height: 1.7em;

  &:hover {
    ${({theme:e})=>l(e,"color","colors.consumption.body.standard.link-hover")};
    text-decoration: underline;
    ${({theme:e})=>l(e,"text-decoration-color","colors.consumption.body.standard.link-hover")};
  }
`;b.defaultProps={colorToken:"colors.consumption.body.standard.body-deemphasized",typeIdentity:"typography.definitions.globalEditorial.context-primary"},e.exports={ChannelCloudContainer:u,ChannelCloudContainerWrapper:d,ChannelCloudHeaderContainer:c,ChannelCloudHeaderLink:g,ChannelCloudHeaderImage:p,ChannelCloudSubChannelContainer:m,ChannelCloudSubChannelLink:y,ChannelCloudSubChannelLinkText:b,ChannelCloudSubChannelText:h}},2669:function(e,t,i){const n=i(2670),{asConfiguredComponent:o}=i(9),{asThemedComponent:a}=i(30);e.exports=a(o(n,"GenericModal"))},2670:function(e,t,i){const{asVariation:n}=i(12),o=i(2671);o.Default=n(o,"Default",{}),e.exports=o},2671:function(e,t,i){const n=i(0),o=i(1),{connect:a}=i(20),r=i(154),l=i(549),s=i(111),{minThresholds:d}=i(17),{GlobalStyle:c,CloseModalButtonDesktop:p,CloseModalButtonMobile:g,ModalContentWrapper:u}=i(2672),m=({children:e,closeModalText:t,isModalOpen:i,openModal:o,modalTransitionTime:a,showHeader:l})=>{n.useEffect(()=>{r.setAppElement("#app-root")},[]);const m=n.useMemo(()=>{var e;if(!l||!i||!document)return 0;const t=null===window||void 0===window?void 0:window.innerWidth,n=document.getElementsByClassName("visual-link-banner--is-scrolled"),o=document.getElementsByClassName("site-navigation");return t<d.xl&&n.length&&n[0]?n[0].offsetHeight:o&&o.length&&(null===(e=o[0])||void 0===e?void 0:e.offsetHeight)||0},[i,l]);return n.createElement(r,{isOpen:i,className:"genericModal",overlayClassName:{base:"genericModalOverlay",afterOpen:"genericModalOverlayAfterOpen",beforeClose:"genericModalOverlayBeforeClose"},bodyOpenClassName:"genericModalBodyOpen",shouldCloseOnEsc:!0,closeTimeoutMS:a},n.createElement(p,{btnStyle:"text",iconPosition:"before",hasEnableIcon:!0,onClickHandler:()=>o(!1),ButtonIcon:s.Close,label:t}),n.createElement(g,{ButtonIcon:s.Close,onClickHandler:()=>o(!1),onTouchStart:()=>o(!1),btnStyle:"outlined",isIconButton:!0,hasEnableIcon:!0,cornerRadius:"FullyRoundedCorner",size:"small",label:t}),n.createElement(u,null,e),n.createElement(c,{siteHeaderHeight:m,modalTransitionTime:a}))};m.propTypes={children:o.node.isRequired,closeModalText:o.string,isModalOpen:o.bool.isRequired,modalTransitionTime:o.number,openModal:o.func.isRequired,showHeader:o.bool},m.defaultProps={closeModalText:"",modalTransitionTime:300,showHeader:!1},e.exports=a(e=>({isModalOpen:e.isModalOpen||!1}),e=>{const{openModal:t}=l(e);return{openModal:t}})(m)},2672:function(e,t,i){const{default:n,createGlobalStyle:o}=i(3),a=i(14),{ButtonLabel:r,ButtonIconWrapper:l}=i(29),{maxScreen:s,minScreen:d,getColorToken:c,calculateSpacing:p,getZIndex:g}=i(4),{maxThresholds:u}=i(17),{BREAKPOINTS:m}=i(6),h=n(a.Utility).withConfig({displayName:"CloseModalButton"})`
  position: absolute;
  color: ${({theme:e})=>c(e,"colors.interactive.base.dark")};

  svg {
    fill: ${({theme:e})=>c(e,"colors.interactive.base.dark")};
  }

  &:hover {
    color: ${({theme:e})=>c(e,"colors.interactive.base.dark")};
  }

  ${l} {
    display: flex;
  }
`,y=n(h).withConfig({displayName:"CloseModalButtonDesktop"})`
  left: 0;

  &:hover {
    text-decoration: underline;
    text-decoration-color: ${c("colors.interactive.base.primary")};
  }

  ${s(u.lg+"px")} {
    display: none;
  }

  ${r} {
    padding: 0;
  }
`,b=n(h).withConfig({displayName:"CloseModalButtonRight"})`
  top: ${p(2.5)};
  right: ${p(3)};
  left: unset;
  z-index: ${g("skipLink")};
  border: 1px solid ${c("colors.interactive.base.light")};
  width: ${p(5)};
  height: ${p(5)};

  &:hover {
    border: 1px solid ${c("colors.interactive.base.light")};
    background: ${c("colors.interactive.base.light")};
  }

  svg {
    vertical-align: bottom;
  }

  ${d(m.lg)} {
    display: none;
  }
`,f=n.div.withConfig({displayName:"ModalContentWrapper"})`
  padding: 0;
  height: 100%;
`,C=o`
  .genericModalBodyOpen {
    overflow: hidden;
  }

  .genericModalOverlay {
    position: fixed;
    top: ${({siteHeaderHeight:e})=>e+"px"};
    left: 0;
    transition: ${({modalTransitionTime:e})=>e&&`opacity ${e}ms ease-in-out;`};
    opacity: 0;
    z-index: ${g("hyperstitialLayer")};
    width: 100%;
    height: ${({siteHeaderHeight:e})=>`calc(100% - ${e}px)`};
  }

  .genericModal {
    background: ${({theme:e})=>c(e,"colors.interactive.base.white")};
    width: 100%;
    height: 100%;
    overflow-y: auto;
  }

  .genericModalOverlayAfterOpen {
    opacity: 1;
  }

  .genericModalOverlayBeforeClose {
    opacity: 0;
  }
`;e.exports={GlobalStyle:C,CloseModalButtonDesktop:y,CloseModalButtonMobile:b,ModalContentWrapper:f}},271:function(e,t,i){var n=this&&this.__rest||function(e,t){var i={};for(var n in e)Object.prototype.hasOwnProperty.call(e,n)&&t.indexOf(n)<0&&(i[n]=e[n]);if(null!=e&&"function"==typeof Object.getOwnPropertySymbols){var o=0;for(n=Object.getOwnPropertySymbols(e);o<n.length;o++)t.indexOf(n[o])<0&&Object.prototype.propertyIsEnumerable.call(e,n[o])&&(i[n[o]]=e[n[o]])}return i};const o=i(0),a=i(1),r=e=>{var{fillColor:t="#000",width:i="100px",height:a="100px",margin:r="20px"}=e,l=n(e,["fillColor","width","height","margin"]);return o.createElement("svg",Object.assign({style:{width:i,height:a,margin:r,display:"inline-block"}},l,{version:"1.1",id:"L9",xmlns:"http://www.w3.org/2000/svg",x:"0px",y:"0px",viewBox:"0 0 100 100",enableBackground:"new 0 0 0 0"}),o.createElement("path",{fill:t,d:"M73,50c0-12.7-10.3-23-23-23S27,37.3,27,50 M30.9,50c0-10.5,8.5-19.1,19.1-19.1S69.1,39.5,69.1,50"},o.createElement("animateTransform",{attributeName:"transform",attributeType:"XML",type:"rotate",dur:"1s",from:"0 50 50",to:"360 50 50",repeatCount:"indefinite"})))};r.propTypes={fillColor:a.string,height:a.string,margin:a.string,width:a.string},e.exports=r},289:function(e,t,i){e.exports=i(454)},308:function(e,t,i){const{fetchWithTimeout:n}=i(54),{loadData:o}=i(453);e.exports={fetchWithTimeout:n,loadData:o}},433:function(e,t,i){const n=i(1474),o=i(15),a=e=>{const t=o(e,"gradient-style"),i=o(e,"color-stops");let n=o(e,"angle");"radial"===t&&(n=null);return`background: ${((e,t,i=null)=>null===i?`${e}-gradient(${t})`:`${e}-gradient(${i}, ${t})`)(t,i,n)}; position: sticky;`},r=e=>{const t=o(e,"url"),i=o(e,"repeat"),n=o(e,"color"),a=o(e,"attachment"),r=o(e,"size");return((e,t="no-repeat",i="center",n="cover",o="scroll",a="transparent")=>{let r,l="";if(null!==e&&(r=e),l=`background-image:${r};\n      background-attachment:${o};\n      background-color:${a};\n      background-repeat:${t};\n      background-position:${i};`,"string"==typeof n)l+=`background-size:${n};`;else{const e=[];n.forEach(t=>{e.push(t)}),l+=`background-size:${n.toString()};`}return l})(t,i,o(e,"position"),r,a,n)},l=e=>{return n(e,"gradient")?a(o(e,"gradient")):n(e,"image")?r(o(e,"image")):n(e,"solid")?(t=o(e,"solid"),`background-color:${o(t,"color")};`):"background: none;";var t};e.exports={getPattern:(e,t)=>{let i;if(!e||0===Object.keys(e).length||!t)return"background: none;";if(n(e,"container-styles")){const s=e["container-styles"];if(!s[t])return"background: none;";i=s[t];const d=n(i,"pattern")?i.pattern:null;if(d&&d.length){let e="";return d.length>=2?(d.forEach((t,i)=>{let l=i;if(n(t,"gradient"))e+=a(o(t,"gradient"));else if(n(t,"image")){let i="";o(t,"image").size&&(i=o(t,"url")),i.size?e+=`'url(${i})'`:l=d.length,r(o(t,"image"))}else if(n(t,"solid")){const i=o(t,"solid");i&&(e+=o(i,"color"))}l!==d.length-1&&(e+=",")}),e.toString()):l(d[0])}}return"background: none;"}}},435:function(e,t,i){const n=i(0),{VogueWrapper:o}=i(436);e.exports=()=>n.createElement(o,{xmlns:"http://www.w3.org/2000/svg",viewBox:"0 0 32 36",textRendering:"geometricPrecision",shapeRendering:"geometricPrecision",className:"loader-vogue"},n.createElement("path",{id:"E",className:"st0",d:"M23.2612,0L23.2612,11.7649C23.2612,11.8148,23.0279,11.8315,23.0279,11.7649C23.0279,11.6982,23.2612,0.333282,12.7129,0.333282L8.76346,0.333282C8.54683,0.333282,8.41352,0.36661,8.41352,0.549915L8.41352,15.231C8.41352,15.3643,8.54683,15.4476,8.76346,15.4476L10.0799,15.4476C16.3456,15.4476,15.8457,9.14859,15.8457,9.08193C15.8457,9.01528,16.0957,9.03194,16.0957,9.08193C16.0957,9.13193,16.1123,22.2966,16.1123,22.3466C16.1123,22.3966,15.929,22.4132,15.929,22.3466C15.929,22.2799,15.779,15.7809,9.94661,15.7809L8.76346,15.7809C8.56349,15.7809,8.41352,15.8476,8.41352,15.9975L8.41352,31.7951C8.41352,31.9118,8.54683,32.0117,8.76346,32.0117C9.64666,32.0117,11.9296,32.0284,12.5629,32.0284C23.8945,32.0617,23.7445,19.8969,23.7445,19.8469C23.7445,19.797,24.0278,19.797,24.0278,19.8469L24.0278,32.445L0.114794,32.445C0.0481373,32.445,0.0481373,32.0951,0.114794,32.0951L2.49776,32.0951C2.71439,32.0951,2.84771,32.0284,2.84771,31.8784C2.84771,29.1455,2.84771,0.799877,2.84771,0.549915C2.84771,0.349946,2.58108,0.333282,2.49776,0.333282C2.46443,0.333282,1.23129,0.333282,0.0148091,0.333282C0.0148091,0.333282,-0.0685114,0.166641,0.0148091,0C0.0981296,0,23.2612,0,23.2612,0",opacity:"0",transform:"translate(4.15223,1.87884)",style:{animation:"E_o 1.5s linear infinite both"}}),n.createElement("path",{id:"U",className:"st0",d:"M27.1719,0.266719C25.7716,0.266719,24.3713,0.266719,24.3713,0.266719C24.3713,0.266719,24.0214,0.266719,24.0214,0.466748C24.0214,2.70041,24.0214,21.5031,24.0214,24.1868L24.0214,24.2034C24.038,30.8378,18.9372,33.0048,15.2701,33.0548L15.2701,33.0381C10.9528,33.0548,3.20169,32.338,3.20169,22.77C3.20169,22.77,3.18502,1.00016,3.16835,0.700115C3.15167,0.233381,2.55159,0.250051,2.55159,0.250051C2.55159,0.250051,0.10124,0.250051,0.017895,0.250051C-0.0654503,0.250051,-0.0487813,-0.1,0.017895,-0.1L11.9529,-0.1C12.0196,-0.1,12.003,0.250051,11.9529,0.250051C11.9029,0.250051,9.41925,0.250051,9.41925,0.250051C9.41925,0.250051,8.6358,0.216712,8.6358,0.733453C8.6358,1.46689,8.65246,25.0203,8.65246,25.8537C8.65246,29.5209,10.7361,32.7714,15.2534,32.7048C18.7873,32.638,23.6879,30.5711,23.6714,24.1868C23.6714,23.8701,23.6714,2.83376,23.6714,0.466748C23.6546,0.283389,23.3046,0.266719,23.3046,0.266719L19.9875,0.266719C19.9207,0.266719,19.9207,-0.0666619,20.004,-0.0666619C20.0208,-0.0666619,27.0885,-0.0666619,27.1719,-0.0666619C27.2385,0.100029,27.1719,0.266719,27.1719,0.266719Z",opacity:"0",transform:"translate(2.57354,1.9455)",style:{animation:"U_o 1.5s linear infinite both"}}),n.createElement("path",{id:"G",className:"st0",d:"M16.5382,19.9408L19.2012,19.9408C19.2012,19.9408,19.7005,19.8909,19.7005,20.124C19.7005,20.1406,19.7005,27.7636,19.7005,27.7636C19.7005,33.6389,13.8419,33.7056,12.1941,33.5224C6.06906,32.8233,5.95255,20.1905,5.91927,16.8784C5.83605,7.85727,7.2508,0.00125154,13.4091,0.317489C20.7159,0.683659,22.63,11.1694,22.7465,11.6355C22.9962,11.7187,22.9962,11.519,22.9962,11.519L23.0127,0.0844721C23.0127,0.0844721,22.9628,-0.0153926,22.813,0.11776C22.7631,0.167692,22.6965,0.23427,22.63,0.300845C19.401,3.59637,17.7866,0.134404,13.0762,-0.0153926C7.28409,-0.198477,0.0938322,7.82398,-0.00603238,17.0947C-0.105897,26.4154,6.01913,33.6056,12.7101,33.8554C16.3385,33.9885,16.2553,32.9898,20.7324,31.5917C23.4621,30.743,24.7604,32.5904,24.8602,33.356C25.0267,33.4725,25.0434,33.2727,25.0434,33.2727L25.0434,20.1739C25.0434,19.9077,25.4094,19.9576,25.4094,19.9576L27.8729,19.9576C27.9394,19.9576,27.9394,19.5914,27.8729,19.5914L16.5216,19.5914C16.4383,19.5914,16.4383,19.9408,16.5382,19.9408",opacity:"0",transform:"translate(2.19725,0.935133)",style:{animation:"G_o 1.5s linear infinite both"}}),n.createElement("path",{id:"O",className:"st0",d:"M5.92208,16.6329C5.92208,7.61677,7.65213,0.247436,14.0399,0.264071C20.9103,0.264071,22.2578,7.93283,22.2578,16.6496C22.2578,25.6659,20.9435,33.4346,14.09,33.4346C6.73721,33.4177,5.92208,26.115,5.92208,16.6329M13.9735,33.6839C20.7938,33.7005,28.1633,26.1649,28.18,16.8159C28.1965,7.01792,21.1101,0.0478142,14.09,-0.00209099C6.90356,-0.0686311,0,7.83302,0,16.8159C0,26.0983,6.52095,33.6507,13.9735,33.6839",opacity:"0",transform:"translate(2.065,1.05937)",style:{animation:"O_o 1.5s linear infinite both"}}),n.createElement("path",{id:"V",className:"st0",d:"M0.0178861,0L11.7176,0C11.7843,0,11.7676,0.349993,11.7176,0.349993L9.31769,0.349993C9.31769,0.349993,8.63437,0.31666,8.85104,0.699985C8.88437,0.783317,17.1175,23.2828,17.1175,23.2828C17.1175,23.2828,17.2675,23.6662,17.3842,23.7162C17.3842,23.7162,24.7007,1.11664,24.7007,1.09998C24.7674,0.883315,25.034,0.366659,24.584,0.366659L21.9341,0.366659C21.8841,0.366659,21.8841,0.0166663,21.9341,0.0166663L28.234,0.0166663C28.284,0.0166663,28.284,0.366659,28.234,0.366659L25.884,0.366659C25.234,0.349993,25.234,0.549989,25.1174,0.883315C25.0674,1.04998,14.9342,32.616,14.9342,32.616C14.9342,32.616,14.8842,32.4827,14.8509,32.3493C10.5677,19.9663,3.93447,2.73328,3.40115,0.883315C3.25115,0.333326,3.16782,0.349993,2.83449,0.349993C2.71783,0.349993,0.101218,0.349993,0.0178861,0.349993C-0.0654455,0.349993,-0.0487792,0,0.0178861,0",opacity:"0",transform:"translate(2.03854,1.8455)",style:{animation:"V_o 1.5s linear infinite both"}}))},436:function(e,t,i){const n=i(3).default.svg.withConfig({displayName:"VogueWrapper"})`
  animation: rotate 2s linear infinite;
  width: 36px;
  height: 32px;

  & .path {
    stroke: #5652bf;
    stroke-linecap: round;
    animation: dash 1.5s ease-in-out infinite;
  }

  @keyframes E_o {
    0% {
      opacity: 0;
    }

    80.5556% {
      animation-timing-function: cubic-bezier(0, 0, 0.58, 1);
      opacity: 0;
    }

    83.3333% {
      opacity: 1;
    }

    97.2222% {
      animation-timing-function: cubic-bezier(0.42, 0, 1, 1);
      opacity: 1;
    }

    100% {
      opacity: 0;
    }
  }
  @keyframes U_o {
    0% {
      opacity: 0;
    }

    63.8889% {
      animation-timing-function: cubic-bezier(0, 0, 0.58, 1);
      opacity: 0;
    }

    66.6667% {
      opacity: 1;
    }

    80.5556% {
      animation-timing-function: cubic-bezier(0.42, 0, 1, 1);
      opacity: 1;
    }

    83.3333% {
      opacity: 0;
    }

    100% {
      opacity: 0;
    }
  }
  @keyframes G_o {
    0% {
      opacity: 0;
    }

    47.2222% {
      animation-timing-function: cubic-bezier(0, 0, 0.58, 1);
      opacity: 0;
    }

    50% {
      opacity: 1;
    }

    63.8889% {
      animation-timing-function: cubic-bezier(0.42, 0, 1, 1);
      opacity: 1;
    }

    66.6667% {
      opacity: 0;
    }

    100% {
      opacity: 0;
    }
  }
  @keyframes O_o {
    0% {
      opacity: 0;
    }

    30.5556% {
      animation-timing-function: cubic-bezier(0, 0, 0.58, 1);
      opacity: 0;
    }

    33.3333% {
      opacity: 1;
    }

    47.2222% {
      animation-timing-function: cubic-bezier(0.42, 0, 1, 1);
      opacity: 1;
    }

    50% {
      opacity: 0;
    }

    100% {
      opacity: 0;
    }
  }
  @keyframes V_o {
    0% {
      opacity: 0;
    }

    13.8889% {
      animation-timing-function: cubic-bezier(0, 0, 0.58, 1);
      opacity: 0;
    }

    16.6667% {
      animation-timing-function: cubic-bezier(0, 0, 0.58, 1);
      opacity: 1;
    }

    30.5556% {
      animation-timing-function: cubic-bezier(0.42, 0, 1, 1);
      opacity: 1;
    }

    33.3333% {
      opacity: 0;
    }

    100% {
      opacity: 0;
    }
  }
`;e.exports={VogueWrapper:n}},448:function(e,t,i){const{asConfiguredComponent:n}=i(9),o=i(449);e.exports=n(o,"ChannelNavigation")},449:function(e,t,i){const n=i(0),{useState:o,useRef:a,useEffect:r}=i(0),l=i(1),{useIntl:s}=i(2),d=i(44),c=i(450).default,{connector:p}=i(18),{trackComponent:g}=i(5),{googleAnalytics:u}=i(13),m=i(451),h=i(126),y=i(32),b=i(107),{INITIAL_STATE:f,computeScroll:C}=i(31),{ChannelNavigationWrapper:v,ChannelNavigationContainer:w,ChannelNavigationContent:S,ChannelNavigationLogoWrapper:k,ChannelNavigationLogo:x,ChannelNavigationScrollViewLogo:$,ChannelNavigationLinksWrapper:T,ChannelNavigationLinksList:E,ChannelNavigationLinkItem:N,ChannelNavigationLink:B,ChannelNavigationChannelDrawer:L,ChannelNavigationGroupedNavigation:I,ChannelNavigationGlobalDrawer:A,ChannelNavigationAccount:P,ChannelNavigationToggle:R,ChannelNavigationSecondaryMenu:D}=i(459),M=({accountProps:e,isFixed:t,logo:i,isAccountsEnabled:l,scrollViewLogo:p,channelNavigationLinks:M,channelNavigationLogoBaseUrl:O,overrideChannelNavigationLinks:H,secondaryMenuProps:_,showExternalProfileLink:G,loaderType:W,onNavigationLinkClick:j,activeLinkIndex:F,user:V,hideLinksOnMobile:U,hideDrawerScroll:z})=>{n.useEffect(()=>{g("ChannelNavigation")},[]);const[q,K]=o(!1),[Z,X]=n.useState(!1),[Y,J]=o(null),[Q,ee]=o(f),te=a(null),{formatMessage:ie}=s(),ne=H||M;n.useEffect(()=>{const e=e=>{"Escape"===e.key&&Z&&(u.emitGoogleTrackingEvent("hamburger-menu-"+(Z?"collapsed":"expanded")),X(!1))};return Z&&window.addEventListener("keyup",e),()=>window.removeEventListener("keyup",e)},[Z]);const oe=()=>{ee(e=>{var t,i;const n=C(e),o=(null===(t=null===document||void 0===document?void 0:document.body)||void 0===t?void 0:t.scrollHeight)-(null===(i=null===document||void 0===document?void 0:document.body)||void 0===i?void 0:i.clientHeight);return Object.assign(Object.assign({},n),{scrollHeight:o})})};r(()=>{const e=d(oe,100);return window.addEventListener("scroll",e,{passive:!0}),()=>window.removeEventListener("scroll",e)},[t]);const{direction:ae,pageYOffset:re,scrollHeight:le}=Q,se=Z?y:b,de=t||re>0,ce=t||"up"!==ae&&re>0||le===re;return ne&&ne.length?n.createElement(v,{isFixed:de},n.createElement(w,{ref:te,"data-testid":"channel-navigation",hideLinksOnMobile:U},n.createElement(S,{isFixed:de,isScrollingDown:ce},i&&p&&n.createElement(k,{isFixed:de,isScrollingDown:ce},n.createElement("a",{href:O},n.createElement(x,Object.assign({isScrollingDown:ce},i)),ce&&n.createElement($,Object.assign({isScrollingDown:ce},p))))),n.createElement(T,{isScrollingDown:ce,hideLinksOnMobile:U},n.createElement(E,{"data-journey-hook":"channel-navigation"},ne.map((e,t)=>{const i=void 0===F||t===F;return n.createElement(N,{key:e.key||e.type},n.createElement(B,{tabIndex:0,isActive:i,label:e.text,href:e.href,as:"a",isInline:!0,onClick:t=>{e.apiEndpoint&&(t.preventDefault(),K(!0),J(Object.assign({},e))),j&&j(e),u.emitGoogleTrackingEvent(e.analyticsEvent)}},e.text))}))),!l&&G&&n.createElement(P,{isScrollingDown:ce,signInLabel:null==G?void 0:G.signInLabel,signInLink:null==G?void 0:G.signInLink,user:{isAuthenticated:!1}}),l&&V&&n.createElement(P,{isScrollingDown:ce,accountLinks:e.accountLinks,savedStoriesLabel:null==e?void 0:e.savedStoriesLabel,accountBookmarkAlertLabel:null==e?void 0:e.accountBookmarkAlertLabel,accountLabel:null==e?void 0:e.accountLabel,signInLabel:null==e?void 0:e.signInLabel,signInLink:null==e?void 0:e.signInLink,signOutLink:null==e?void 0:e.signOutLink,user:V,className:"standard-navigation__section--utility-links-login"}),n.createElement(R,{tabIndex:0,isIconButton:!0,isScrollingDown:ce,ButtonIcon:se,label:"Open Navigation Menu",onClickHandler:()=>{u.emitGoogleTrackingEvent("hamburger-menu-"+(Z?"collapsed":"expanded")),X(!Z)},role:"button","aria-expanded":Z})),!!Y&&n.createElement(L,{isOpen:q,onClose:()=>{K(!1)},hideDrawerScroll:z,contentLabel:ie(c.channelDrawerContentLabel),showCloseButton:!0,className:"channel-navigation-drawer"},n.createElement(I,null,n.createElement(m,{storeKey:Y.key,dataUrl:Y.apiEndpoint,hasAtoZIndex:Y.hasAtoZIndex,loaderType:W,hasFilter:Y.hasFilter,filterLabel:Y.filterLabel}))),n.createElement(A,{isOpen:Z,onClose:()=>ee(!Z),contentLabel:"Navigation Menu"},n.createElement(D,{isFixed:de},!l&&G?n.createElement(h,Object.assign({},_,{user:{isAuthenticated:!1},isAccountsEnabled:!0,contentAlign:"center"})):n.createElement(h,Object.assign({accountProps:e},_,{user:V,isAccountsEnabled:l,contentAlign:"center"}))))):null},O=l.shape({text:l.string,key:l.string,apiEndpoint:l.string});M.defaultProps={accountProps:{accountLinks:[]},hideLinksOnMobile:!1,isAccountsEnabled:!1},M.propTypes={accountProps:l.object,activeLinkIndex:l.number,channelNavigationLinks:l.arrayOf(O),channelNavigationLogoBaseUrl:l.string,hideDrawerScroll:l.bool,hideLinksOnMobile:l.bool,isAccountsEnabled:l.bool,isFixed:l.bool,loaderType:l.string,logo:l.object,onNavigationLinkClick:l.func,overrideChannelNavigationLinks:l.arrayOf(O),scrollViewLogo:l.object,secondaryMenuProps:l.object,showExternalProfileLink:l.object,user:l.shape({isAuthenticated:l.bool.isRequired})},M.displayName="ChannelNavigation",e.exports=p(M,{keysToPluck:["user","isAccountsEnabled","accountProps"]})},450:function(e,t,i){Object.defineProperty(t,"__esModule",{value:!0});const n=i(2);t.default=n.defineMessages({toggleLabel:{id:"ChannelNavigation.ToggleLabel",defaultMessage:"Open Navigation Menu",description:"ChannelNavigation component toggle label"},channelDrawerContentLabel:{id:"ChannelNavigation.ChannelDrawerContentLabel",defaultMessage:"Runway filters navigation",description:"ChannelNavigation component channel drawer content label"},globalDrawerContentLabel:{id:"ChannelNavigation.GlobalDrawerContentLabel",defaultMessage:"Navigation Menu",description:"ChannelNavigation component global drawer content label"}})},451:function(e,t,i){e.exports=i(452)},452:function(e,t,i){var n=this&&this.__rest||function(e,t){var i={};for(var n in e)Object.prototype.hasOwnProperty.call(e,n)&&t.indexOf(n)<0&&(i[n]=e[n]);if(null!=e&&"function"==typeof Object.getOwnPropertySymbols){var o=0;for(n=Object.getOwnPropertySymbols(e);o<n.length;o++)t.indexOf(n[o])<0&&Object.prototype.propertyIsEnumerable.call(e,n[o])&&(i[n[o]]=e[n[o]])}return i};const o=i(1),a=i(0),{useState:r,useEffect:l}=i(0),{connect:s}=i(20),{loadData:d}=i(308),c=i(289),p=i(263),{LoaderWrapper:g}=i(458),u=i(155),{trackComponent:m}=i(5),h=e=>{var{dataUrl:t,storeKey:i,data:o,setData:s,loaderType:h="Circle"}=e,y=n(e,["dataUrl","storeKey","data","setData","loaderType"]);a.useEffect(()=>{m("GroupedNavigationContainer")},[]);const[b,f]=r(!1),[C,v]=r(!1),w=p[h];return l(()=>{!async function(){if(!o&&t){f(!0);try{const e=await d({url:""+t}),n=e?e[i]:[];s(i,n)}catch(e){v(!0)}finally{f(!1)}}}()},[t]),a.createElement(a.Fragment,null,C&&a.createElement(u.ContentCenterNoBackground,{ariaLive:"polite",className:"brand-background__lede",dangerousHed:"Oops",dangerousDek:"something went wrong"}),b&&a.createElement(g,null,a.createElement(w,null)),o&&a.createElement(c,Object.assign({groupedLinks:o},y)))};h.propTypes={analyticsEventForFilter:o.string,data:o.array,dataUrl:o.string.isRequired,filterLabel:o.string,hasAtoZIndex:o.bool,hasFilter:o.bool,loaderType:o.string,setData:o.func.isRequired,storeKey:o.string.isRequired},e.exports=s((e,{storeKey:t})=>{var i;return{data:(null===(i=e.groupedNavigation)||void 0===i?void 0:i[t])||null}},e=>({setData:(t,i)=>{e({type:"MERGE_KEY",key:"groupedNavigation",value:{[t]:i}})}}))(h)},453:function(e,t){e.exports={loadData:async function({url:e,gtmEvent:t=null}){window.dataLayer&&t&&window.dataLayer.push({event:t});const i=await fetch(e);if(i.ok)return i.json();throw new Error(i.statusText)}}},454:function(e,t,i){const n=i(1),o=i(0),{useIntl:a}=i(2),{useState:r}=i(0),l=i(133),s=i(86),d=i(70),{asConfiguredComponent:c}=i(9),{filter:p,useDebounce:g}=i(455),u=i(456),{trackComponent:m}=i(5),{GroupedNavigationWrapper:h,GroupedNavigationFilter:y,GroupedNavigationFilterContent:b,GroupedNavigationFilterInput:f,GroupedNavigationContent:C,GroupedNavigationLinks:v,GroupedNavigationGroup:w,GroupedNavigationIndex:S}=i(180),k=i(457).default,x=({className:e,groupedLinks:t,showContentDivider:i=!0,hasAtoZIndex:n=!1,hasFilter:c=!1,analyticsEventForFilter:x,filterLabel:$})=>{o.useEffect(()=>{m("GroupedNavigation")},[]);const{formatMessage:T}=a(),E=o.useRef(null),N=l(),[B,L]=r(""),[I,A]=g(t,200);return t&&t.length?o.createElement(h,{className:e,hasFilter:c,"data-testid":"GroupedNavigationWrapper"},c&&o.createElement(y,null,o.createElement(b,null,o.createElement(f,{placeholder:$,"aria-label":$||T(k.filterInputAriaLabelText),name:"filter",type:"text",onChange:e=>{const i=e.target.value;L(i),A(()=>p(t,i))},onFocus:()=>{x&&s.emitGoogleTrackingEvent(x)},value:B}),o.createElement(d,null))),o.createElement(C,{hasFilter:c},o.createElement(v,{ref:E},I.map(e=>{if(!e.links)return null;const t=e.links.map(e=>{const t=!0===e.isSecondary?"link--secondary":"link--primary";return Object.assign(Object.assign({},e),{className:t})}),a={};return n&&(a.id="#"===e.groupName?"other-"+N:`${e.groupName.toLowerCase()}-${N}`),o.createElement(w,{key:e.groupName,className:"grouped-navigation__group",links:t,linkClassName:"grouped-navigation__link",heading:e.groupName,showContentDivider:i,shouldStyleListItems:!0,attributes:a})})),n&&o.createElement(S,{className:"grouped-navigation__index"},o.createElement(u,{links:t,linksRef:E,navId:N})))):null},$=n.shape({text:n.string.isRequired,url:n.string.isRequired,isSecondary:n.bool,analyticsEvent:n.string}),T=n.arrayOf(n.shape({links:n.arrayOf($),groupName:n.string,groupId:n.string}));x.propTypes={analyticsEventForFilter:n.string,className:n.string,filterLabel:n.string,groupedLinks:T,hasAtoZIndex:n.bool,hasFilter:n.bool,showContentDivider:n.bool},x.displayName="GroupedNavigation",e.exports=c(x,"GroupedNavigation"),e.exports.groupedLinksShape=T},455:function(e,t,i){Object.defineProperty(t,"__esModule",{value:!0}),t.useDebounce=t.filter=void 0;const{useState:n,useCallback:o}=i(0),a=i(45);t.filter=(e,t)=>{if(!(null==t?void 0:t.trim()))return e;const i=e=>e.toString().toLowerCase().normalize("NFD").replace(/[\u0300-\u036f]/g,"").replace(/[^a-z0-9\s]/gi,"");return e.map(e=>{var n;return Object.assign(Object.assign({},e),{links:null===(n=e.links)||void 0===n?void 0:n.filter(e=>/(\s|')/.test(t)?i(e.text).includes(i(t)):e.text.match(/([a-zA-Z]\.){2,}/)?e.text.split(" ").filter(e=>i(e).startsWith(i(t))).length:e.text.split(/([ \-'’.]+)/).filter(e=>i(e).startsWith(i(t))).length)})}).filter(e=>{var t;return null===(t=e.links)||void 0===t?void 0:t.length})};t.useDebounce=(e,t)=>{const[i,r]=n(e),l=o(a(e=>{r(e)},t),[]);return[i,e=>{l(e)}]}},456:function(e,t,i){const n=i(1),o=i(0),{AtoZIndexWrapper:a,AtoZIndexList:r,AtoZIndexLink:l,AtoZIndexText:s}=i(180),d=["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z","#"],c=({links:e,linksRef:t,navId:i})=>{if(!e||!e.length)return null;const n=e=>{e.preventDefault();const i=window.requestAnimationFrame||(e=>window.setTimeout(e,20)),n=document.getElementById(e.target.hash.replace("#",""));if(!n)return;const{offsetTop:o}=n,{offsetTop:a,scrollTop:r}=t.current,l=o-a-r;let s=0;const d=()=>{s+=20;const e=function(e,t,i,n){let o=e;return o/=n/2,o<1?i/2*o*o+t:(o--,-i/2*(o*(o-2)-1)+t)}(s,r,l,600);t.current.scrollTop=e,s<600&&i(d)};d()};return o.createElement(a,{"data-testid":"AtoZIndexWrapper"},o.createElement(r,null,d.map(t=>{const a=e.find(e=>e.groupName===t);return a?o.createElement("li",{key:t},o.createElement(l,{"data-testid":"AtoZIndexLink",href:"#"+("#"===a.groupName?"other-"+i:`${a.groupName.toLowerCase()}-${i}`),onClick:n},t)):o.createElement(s,{key:t},t)})))};c.propTypes={links:n.arrayOf(n.shape({groupName:n.string.isRequired})),linksRef:n.object,navId:n.string},e.exports=c},457:function(e,t,i){Object.defineProperty(t,"__esModule",{value:!0});const n=i(2);t.default=n.defineMessages({filterInputAriaLabelText:{id:"GroupedNavigation.FilterInputAriaLabel",defaultMessage:"Filter links",description:"Grouped Navigation Filter component aria label text",isConfigurable:!0}})},458:function(e,t,i){const n=i(3).default,{calculateSpacing:o}=i(4),a=n.div.withConfig({displayName:"LoaderWrapper"})`
  padding-top: ${o(6)};
  text-align: center;
`;e.exports={LoaderWrapper:a}},459:function(e,t,i){const n=i(3).default,{calculateSpacing:o,getColorToken:a,getTypographyStyles:r,getZIndex:l}=i(4),{hideVisually:s}=i(51),{BREAKPOINTS:d,ZINDEX_MAP:c}=i(6),{maxThresholds:p}=i(17),g=i(14),u=i(84),m=i(49),h=i(19),y=i(173),{SecondaryMenuAccount:b}=i(171),{StandardNavigationDropdown:f,StandardNavigationAccountLabel:C,AccountDropdownToggleIcon:v}=i(87),{GridItem:w}=i(25),{SignOutButtonWrapper:S}=i(124),k=n.nav.withConfig({displayName:"ChannelNavigationWrapper"})`
  position: relative;
  z-index: ${c.persistentTopLayer};
  max-height: ${o(24)};
  ${({isFixed:e})=>e&&"\n      position: fixed;\n      top: 0;\n      right: 0;\n      left: 0;\n      "};
`;k.displayName="ChannelNavigationWrapper";const x=n.div.withConfig({displayName:"ChannelNavigationContainer"})`
  position: relative;
  ${({hideLinksOnMobile:e})=>`padding-bottom: ${o(e?0:7)};`}

  @media (min-width: ${d.md}) {
    border-bottom: 1px solid rgba(51, 51, 51, 1);
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    background: ${({theme:e})=>a(e,"colors.consumption.lead.inverted.background")};
    padding-bottom: 0;
  }
`,$=n.div.withConfig({displayName:"ChannelNavigationContent"})`
  display: flex;
  position: relative;
  flex-wrap: wrap;
  z-index: 1;
  margin: 0 auto;
  border-bottom: 1px solid
    ${({theme:e})=>a(e,"colors.consumption.lead.inverted.divider")};
  background: ${({theme:e})=>a(e,"colors.consumption.lead.inverted.background")};
  @media (min-width: ${d.md}) {
    flex-wrap: ${({isScrollingDown:e})=>e?"nowrap":"wrap"};
  }
`,T=n.div.withConfig({displayName:"ChannelNavigationLogoWrapper"})`
  display: flex;
  justify-content: center;
  margin: 0 auto;
  padding: ${o(1)};

  @media (min-width: ${d.md}) {
    ${({isScrollingDown:e})=>e?`\n      left: ${o(3)};\n      position: absolute;\n      padding: ${o(1)};\n      width: unset;\n    `:`\n      width:100vw; \n      padding: ${o(1)} 0;\n    `}
  }
`,E=n(m).withConfig({displayName:"ChannelNavigationLogo"})`
  width: 96px;
  @media (min-width: ${d.md}) {
    display: flex;
    padding: ${o(1)} 0;
    width: 168px;
    height: 88px;
    ${({isScrollingDown:e})=>e&&`\n        ${s()}\n      `};
  }
`,N=n(m).withConfig({displayName:"ChannelNavigationScrollViewLogo"})`
  @media (max-width: ${d.md}) {
    ${s()}
  }
  padding: ${o(.5)} 0;
  width: 83px;
  height: unset;
`,B=n(h.NoMargins).withConfig({displayName:"ChannelNavigationLinksWrapper"})`
  > ${w} {
    grid-column: 1;
    margin: 0 auto;
    text-align: center;
    @media (min-width: ${d.md}) {
      grid-column: 2 / span 10;
    }
  }

  position: absolute;
  top: auto;
  transition: transform 0.5s ease-in-out;
  background: ${({theme:e})=>a(e,"colors.consumption.lead.inverted.background")};
  width: 100%;
  overflow-x: auto;
  overflow-y: hidden;

  ${({isScrollingDown:e})=>e?"transform: translateY(-100%);":"transform: translateY(0%);"}

  @media (min-width: ${d.md}) {
    display: grid;
    position: initial;
    align-items: center;
    justify-content: center;
    transform: none;
    margin: 0 calculateSpacing(14.5);
    height: 64px;
  }

  @media (max-width: ${p.md}px) {
    ${({hideLinksOnMobile:e})=>e?s()+";":`padding: ${o(2)} 0 ${o(2)}\n      ${o(3)};\n      &::after {\n        background: linear-gradient(\n          to right,\n          rgba(0, 0, 0, 0.01) 31.25%,\n          ${({theme:e})=>a(e,"colors.consumption.lead.inverted.background")}\n            100%\n        );\n        content: '';\n        height: ${o(4)};\n        width: 48px;\n        bottom: ${o(2)};\n        right: 0;\n        position: sticky;\n        pointer-events: none;`}
  }
`,L=n.ul.withConfig({displayName:"ChannelNavigationLinksList"})`
  display: flex;
  margin: 0 auto;
  list-style: none;
  text-align: center;
  padding-inline-start: 0;

  @media (max-width: ${d.md}) {
    ${({hideLinksOnMobile:e})=>e?s()+";":""}
  }
`,I=n.li.withConfig({displayName:"ChannelNavigationLinkItem"})`
  padding-right: ${o(2)};

  &:last-child {
    padding-right: 0;
  }

  @media (min-width: ${d.md}) {
    margin-right: 0;
    padding-right: ${o(3)};
  }
`,A=n.a.withConfig({displayName:"ChannelNavigationLink"})`
  ${({theme:e})=>r(e,"typography.definitions.foundation.link-primary")}

  border: none;
  min-width: auto;
  text-decoration: none;
  white-space: nowrap;
  color: rgb(
    ${({theme:e})=>a(e,"colors.consumption.lead.inverted.link",{rgbOnly:!0})},
    ${({isActive:e})=>e?"1":"0.6"}
  );

  &:hover {
    color: ${({theme:e})=>a(e,"colors.consumption.lead.inverted.link-hover")};
  }
`,P=n(u.Right).withConfig({displayName:"ChannelNavigationChannelDrawer"})`
  &&& {
    ${({hideDrawerScroll:e})=>e&&"\n        height: 100%;\n    "}
  }
  @media (min-width: ${d.md}) {
    width: 400px;
    max-width: 400px;

    && {
      height: 100%;
    }
  }
`,R=n.div.withConfig({displayName:"ChannelNavigationGroupedNavigation"})`
  padding: 0 ${o(4)} ${o(2)};
  height: 100%;
`,D=n(u).withConfig({displayName:"ChannelNavigationGlobalDrawer"})`
  height: auto;
`,M=n(y).withConfig({displayName:"ChannelNavigationAccount"})`
  position: absolute;
  right: ${o(3)};
  border: none;
  background: transparent;
  padding: ${o(1)} ${o(1.5)};

  @media (max-width: ${d.md}) {
    display: none;
  }

  @media (min-width: ${d.md}) {
    left: inherit;
    padding: 0;
    min-width: auto;
  }

  &&&.standard-navigation-account {
    position: absolute;
    top: ${({isScrollingDown:e})=>o(e?1.4:14.4)};
    margin-right: ${o(3)};
    margin-left: ${o(1.5)};
    width: ${o(12)};
    height: ${o(6)};
    white-space: nowrap;
  }

  ${C} {
    justify-content: center;
    color: ${({theme:e})=>a(e,"colors.interactive.base.white")};

    &:hover,
    &:link,
    &:visited,
    &:active {
      color: ${({theme:e})=>a(e,"colors.interactive.base.white")};

      svg {
        path {
          fill: ${({theme:e})=>a(e,"colors.interactive.base.white")};
        }
      }
    }
  }

  .standard-navigation-account--icon,
  ${v} {
    margin-right: ${o(2)};
  }

  ${f} {
    top: ${o(6)};
    background-color: ${({theme:e})=>a(e,"colors.consumption.lead.inverted.background")};
    color: ${({theme:e})=>a(e,"colors.consumption.lead.inverted.link")};

    &::before {
      border-color: black;
    }
  }

  ${f} .account-links__navigation {
    background-color: ${({theme:e})=>a(e,"colors.consumption.lead.inverted.background")};

    .navigation__list-item {
      &:hover {
        background-color: ${({theme:e})=>a(e,"colors.consumption.lead.inverted.background")};
      }

      .navigation__link {
        ${({theme:e})=>r(e,"typography.definitions.foundation.link-secondary")};
        color: ${({theme:e})=>a(e,"colors.consumption.lead.inverted.link")};
      }
    }
  }

  ${f}

  ${S}.account-links__sign-out {
    ${({theme:e})=>r(e,"typography.definitions.foundation.link-secondary")};
    color: ${a("colors.consumption.lead.inverted.link")};

    &:hover {
      background-color: ${a("colors.consumption.lead.standard.divider")};
      color: ${a("colors.consumption.lead.standard.link")};
    }
  }
`,O=n(g.Utility).withConfig({displayName:"ChannelNavigationToggle"})`
  position: absolute;
  top: 14px;
  right: ${o(2)};
  z-index: ${l("dropdown")};
  border: none;
  background: transparent;
  padding: ${o(1)} ${o(1.5)};
  width: ${o(4)};
  height: ${o(4)};
  @media (min-width: ${d.md}) {
    top: ${({isScrollingDown:e})=>e?o(2.5):"122px"};
    left: inherit;
    padding: 0;
    min-width: auto;
  }

  & > div {
    position: absolute;
  }

  path {
    fill: ${({theme:e})=>a(e,"colors.consumption.lead.inverted.link")};
  }

  &:focus {
    outline: 2px auto -webkit-focus-ring-color;
    border: unset;
    background: transparent;
  }

  &:hover {
    border-width: unset;
    border-style: none;
    border-color: transparent;
    background: transparent;

    path {
      fill: ${({theme:e})=>a(e,"colors.consumption.lead.inverted.link-hover")};
    }
  }
`,H=n.div.withConfig({displayName:"ChannelNavigationSecondaryMenu"})`
  padding-top: ${({isFixed:e})=>e?0:"62px"};
  height: 100%;
  ${b} {
    display: block;
  }

  @media (min-width: ${d.md}) {
    padding-top: ${({isFixed:e})=>e?0:"160px"};
  }

  @media (min-width: ${d.lg}) {
    height: 100vh;
    ${b} {
      display: none;
    }
  }
`;e.exports={ChannelNavigationWrapper:k,ChannelNavigationContainer:x,ChannelNavigationContent:$,ChannelNavigationLogoWrapper:T,ChannelNavigationLogo:E,ChannelNavigationScrollViewLogo:N,ChannelNavigationLinksList:L,ChannelNavigationLinksWrapper:B,ChannelNavigationLinkItem:I,ChannelNavigationLink:A,ChannelNavigationChannelDrawer:P,ChannelNavigationGlobalDrawer:D,ChannelNavigationGroupedNavigation:R,ChannelNavigationAccount:M,ChannelNavigationToggle:O,ChannelNavigationSecondaryMenu:H}},470:function(e,t,i){const{asConfiguredComponent:n}=i(9),o=i(532);e.exports=n(o,"BlockquoteEmbed")},471:function(e,t,i){const{default:n}=i(3),o=i(39),{BREAKPOINTS:a,GRID_GAP:r}=i(6),{applyGridSpacing:l,cssVariablesGrid:s}=i(16),{calculateSpacing:d,minMaxScreen:c,getColorToken:p,minScreen:g}=i(4),{SummaryListWrapper:u}=i(161),m=n.div.withConfig({displayName:"SummaryRiverWrapper"})`
  ${s()}

  ${({isFullBleedMobile:e})=>e&&`\n        ${u} {\n            ${c(0,a.md)} {\n              padding: 0;\n            }\n        }\n    `};
  ${({gridColSpanValue:e,showRecircMostPopularInAsideWithRail:t})=>e>=1&&t?`\n           @media (min-width: calc(${a.lg} - 1px)) {\n                .summary-list  .grid-layout__content {\n                  grid-column: span ${e};\n                }\n              }\n            `:""}

  ${({topSpacingInRem:e})=>e?`\n        ${g(a.md)} {\n          margin-top: ${d(e)};\n        }\n      `:""}
`,h=n(o).withConfig({displayName:"SummaryRiverAd"})`
  margin-bottom: ${d(4)};
`,y=n.div.withConfig({displayName:"SummaryRiverTitleWrapper"})`
  ${l("padding")}
  ${({hasScrollOffset:e})=>e?`scroll-margin-top: ${d(8)};`:""}

  margin-bottom: ${d(4)};

  ${({hasExtraTitlePadding:e})=>e?`\n      @media (min-width: ${a.xxl}) {\n        padding-left: calc(2.5 * var(--grid-margin));\n        padding-right: calc(2.5 * var(--grid-margin));\n      }\n      `:""}

  ${({hasDividerAbove:e})=>e?"":`margin-top: ${d(2)};`}
`,b=n.section.withConfig({displayName:"SummaryRiverSection"})``,f=n.div.withConfig({displayName:"SummaryRiverList"})`
  ${({hasRule:e,theme:t})=>e?`\n        &::before {\n          border-top: 1px solid ${p(t,"colors.discovery.body.white.divider")};\n          content: '';\n          grid-column: 1/-1;\n          margin-bottom: ${d(5-r.md)};\n        }\n      `:""}
`;e.exports={SummaryRiverList:f,SummaryRiverWrapper:m,SummaryRiverAd:h,SummaryRiverSection:b,SummaryRiverTitleWrapper:y}},489:function(e,t,i){e.exports=i(715)},498:function(e,t,i){const n=i(3).default,o=i(14),{CaptionWrapper:a,CaptionText:r}=i(59),{calculateSpacing:l,getColorToken:s,getLinkStyles:d}=i(4),c=i(445),{BREAKPOINTS:p}=i(6),{ResponsiveImageContainer:g}=i(24),u=n.div.withConfig({displayName:"LightboxWrapper"})`
  grid-column-start: main;
`,m=n(c).withConfig({displayName:"LightboxSwipe"})`
  display: flex;
  width: 100%;
  height: 100%;
`,h=n(o.Utility).withConfig({displayName:"LightboxCloseButtonIcon"})`
  position: fixed;
  top: ${l(.5)};
  right: ${l(.5)};
  z-index: 1;
  cursor: pointer;
  padding: 8px;
  line-height: 0;

  &.listicle-variation-close {
    top: 1px;
  }

  &,
  &:hover {
    border: 1px solid
      ${({theme:e})=>s(e,"colors.interactive.base.white")};
    background-color: ${({theme:e})=>s(e,"colors.interactive.base.white")};
  }

  &:focus {
    border: 1px solid
      ${({theme:e})=>s(e,"colors.interactive.base.brand-primary")};
    background-color: ${({theme:e})=>s(e,"colors.interactive.base.white")};
  }

  .icon {
    fill: ${({theme:e})=>s(e,"colors.interactive.base.dark")};
  }

  .icon:hover {
    fill: ${({theme:e})=>s(e,"colors.interactive.base.brand-primary")};
    border: 1px solid
      ${({theme:e})=>s(e,"colors.interactive.base.white")};
  }

  @media (min-width: ${p.md}) {
    top: ${l(2)};
    right: ${l(2)};
  }
`,y=n.div.withConfig({displayName:"LightboxSlidesWrapper"})`
  display: flex;
  flex-direction: row;
  flex-wrap: nowrap;
  transition: transform 0.4s ease-in-out;
  height: 100%;

  &.listicle-variation-slide-wrapper {
    display: block;
  }
`,b=n.div.withConfig({displayName:"LightboxSlideTopSpacer"})``,f=n.div.withConfig({displayName:"LightboxSlideImageCaptionOuter"})``,C=n.div.withConfig({displayName:"LightboxSlideCaptionContainer"})``,v=n.div.withConfig({displayName:"LightboxSlideWrapper"})`
  background-color: ${({theme:e})=>s(e,"colors.consumption.lead.standard.background")};
  width: 100vw;

  &.listicle-variation-slide {
    background-color: ${({theme:e})=>s(e,"colors.interactive.base.white")};
    padding: ${l(3)} ${l(9)};

    ${a} {
      margin-bottom: 0;
      background-color: ${({theme:e})=>s(e,"colors.interactive.base.white")};
    }
  }

  ${a} {
    background-color: ${({theme:e})=>s(e,"colors.consumption.lead.standard.background")};
    text-align: initial;

    ${r} {
      ${({theme:e})=>d(e,"colors.consumption.lead.standard.description",null)}

      &:hover {
        text-decoration: none;
      }
    }
  }

  ${({screenOrientation:e})=>"landscape"===e||"portrait"===e||"square"===e?`\n    display: flex;\n    flex-direction: column;\n    justify-content: space-between;\n    height: auto;\n\n    ${f} {\n      display: contents;\n    }\n\n    .responsive-image {\n      display: flex;\n      flex-direction: column;\n\n      ${g} {\n        max-height: 85vh;\n        object-fit: contain;\n      }\n    }\n\n    ${C} {\n      display: flex;\n      flex-direction: column;\n      justify-content: flex-end;\n    }\n\n    ${a} {\n      margin: ${l(2)} ${l(2)} ${l(5)};\n    }\n\n    @media (min-width: ${p.md}) {\n      display: flex;\n      align-items: center;\n      justify-content: center;\n      padding: ${l(4)};\n      height: initial;\n\n      ${b} {\n        display: none;\n      }\n\n      ${f} {\n        display: block;\n      }\n\n      .responsive-asset {\n        display: table-cell;\n      }\n\n      ${C} {\n        display: table-caption;\n        caption-side: bottom;\n      }\n\n      ${a} {\n        margin: 0;\n        margin-top: ${l(1)};\n      }\n    }\n\n    ${"portrait"===e?`\n      @media (min-width: ${p.lg}) {\n        display: flex;\n        flex-direction: row;\n        height: 100%;\n\n        ${b} {\n          display: none;\n        }\n\n        ${f} {\n          display: contents;\n        }\n\n        .responsive-asset {\n          display: flex;\n          height: 100%;\n\n          .responsive-image {\n            height: 100%;\n\n            ${g} {\n              height: 100%;\n              max-height: initial;\n            }\n          }\n        }\n\n        ${C} {\n          display: flex;\n          flex-direction: column;\n          align-self: flex-end;\n        }\n\n        ${a} {\n          margin: 0;\n          margin-bottom: ${l(6)};\n          margin-left: ${l(2)};\n          max-width: 180px;\n        }\n      }\n    `:""}\n  `:""}
`;e.exports={LightboxSwipe:m,LightboxCloseButtonIcon:h,LightboxSlideImageCaptionOuter:f,LightboxSlideCaptionContainer:C,LightboxWrapper:u,LightboxSlideTopSpacer:b,LightboxSlidesWrapper:y,LightboxSlideWrapper:v}},499:function(e,t,i){const n=i(1),o=i(0),{ContentHeaderBylines:a}=i(251),r=({bylineVariation:e,contributors:t})=>t&&0!==Object.keys(t).length?o.createElement(a,{contributors:t,bylineVariation:e,isCompact:!1}):null;r.propTypes={bylineVariation:n.string,contributors:n.object},e.exports=r},500:function(e,t,i){const n=i(1),o=i(0),{ContentHeaderTitleBlockPublishDate:a}=i(176),r=({hasExtraSpaceBetweenSeparator:e,hidePublishDate:t,publishDate:i})=>t||!i?null:o.createElement(a,{hasExtraSpaceBetweenSeparator:e,"data-testid":"ContentHeaderPublishDate"},i);r.propTypes={hasExtraSpaceBetweenSeparator:n.bool,hidePublishDate:n.bool,publishDate:n.string},e.exports=r},501:function(e,t,i){const n=i(1),o=i(0),a=i(1450),r=({hasCategoryEyebrow:e,tags:t,title:i})=>e&&(null==t?void 0:t.length)>0?o.createElement(a,{title:i,tags:t}):null;r.propTypes={hasCategoryEyebrow:n.bool,tags:n.array,title:n.string},e.exports=r},502:function(e,t){e.exports=e=>{if(!window)return{};const{bottom:t,left:i,right:n,top:o}=e.getBoundingClientRect(),a=e.currentStyle||window.getComputedStyle(e);return{bottom:t+parseFloat(a.marginBottom),left:i-parseFloat(a.marginLeft),right:n+parseFloat(a.marginRight),top:o-parseFloat(a.marginTop)}}},503:function(e,t,i){const n=i(1),o=i(0),{SponsorContentContainer:a,SponsorImage:r,SponsoredContent:l,SponsoredContentCampaignLink:s}=i(1457),d=i(11),c=({sponsorByline:e,sponsoredContentHeaderProps:t,theme:i})=>{const{sponsorLogo:n,sponsorName:d,campaignUrl:c}=t;if(!d||!e)return null;const p=`${e} ${d}`;return o.createElement(a,null,o.createElement(s,{additionalRelVals:["sponsored"],href:c},o.createElement(r,Object.assign({},n)),o.createElement(l,{containerTheme:i},p)))};c.propTypes={sponsorByline:n.string,sponsoredContentHeaderProps:n.shape({campaignUrl:n.string,sponsorLogo:n.shape(d.propTypes),sponsorName:n.string}),theme:n.oneOf(["standard","inverted","special"])},c.defaultProps={theme:"inverted"},e.exports=c},504:function(e,t,i){const{default:n}=i(3),{getColorToken:o,getTypographyStyles:a,calculateSpacing:r,maxScreen:l}=i(4),{SocialIconsList:s}=i(28),{BREAKPOINTS:d}=i(6),{BaseText:c,BaseLink:p}=i(10),{SITE_HEADER_TOP_HEIGHT:g,SITE_HEADER_TOP_STICKY_HEIGHT_MD:u,SITE_HEADER_TOP_STICKY_HEIGHT_LG:m}=i(157),h=i(19),{GridItem:y}=i(25),{universalGridCore:b}=i(57),{applyGridSpacing:f}=i(16),C=i(11),v=n.header.withConfig({displayName:"TextOverlayWrapper"})`
  .responsive-clip {
    height: 100%;
  }
  overflow: hidden;
`,w=n.div.withConfig({displayName:"ImageWrapper"})`
  display: flex;
  position: relative;
  align-items: flex-end;
  justify-content: ${({contentAlign:e})=>e};

  @media (orientation: landscape) {
    display: grid;
    min-height: 400px;
  }

  @media (max-width: ${d.md}) {
    display: grid;
    min-height: 667px;
  }

  @media (min-width: ${d.md}) {
    display: grid;
    height: calc(
      100vh - ${g} - ${u}
    );
  }

  @media (min-width: ${d.lg}) {
    display: grid;
    height: calc(
      100vh - ${g} - ${m}
    );
  }

  @media (min-width: ${d.xl}) {
    display: grid;
    min-height: 720px;
  }

  &::after {
    position: absolute;
    top: 0;
    right: 0;
    bottom: 0;
    left: 0;
    background: ${({background:e})=>"gradient"===e?"linear-gradient(to top, rgb(0, 0, 0) 0, transparent 65%)":"rgba(0, 0, 0, 0.65)"};
    content: '';
    pointer-events: none;
  }
`,S=n.div.withConfig({displayName:"Image"})`
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  overflow: hidden;

  .responsive-asset {
    &::before {
      display: block;
      width: 100%;
      content: '';
    }
  }

  > *,
  picture,
  .responsive-asset picture, /* set to override the css specifity set on this component  */
  img {
    object-fit: cover;
    object-position: top;
    width: 100%;
    height: 100%;
  }

  picture {
    position: absolute;
    top: 0;
    left: 0;
    height: 100%;
  }
`,k=n.div.withConfig({displayName:"ContentAlign"})`
  display: flex;
  flex-wrap: wrap;
  justify-content: ${({contentAlign:e})=>e};
  width: 100%;

  > .social-icons--footer {
    ${s} {
      @media (min-width: ${d.md}) {
        position: relative; /* for bookmark onboarding alert placement */
      }
    }
  }

  ${({socialIconsToHide:e})=>e&&`\n      ${l(d.lg)} {\n        ${(e=>e.map(e=>`\n        .social-icons__list-item--${e} {\n          display: none;\n        }\n    `).join("\n"))(e)}\n      }\n    `}

  ${({bottomSpacing:e})=>e&&"margin-bottom: "+r(e)}
`,x=n.div.withConfig({displayName:"Content"})`
  position: relative;
  z-index: 2;
`,$=n(c).withConfig({displayName:"Hed"})`
  text-align: ${({contentAlign:e})=>e};
`;$.defaultProps={as:"h1",colorToken:"colors.consumption.lead.inverted.heading",topSpacing:2,typeIdentity:"typography.definitions.consumptionEditorial.hed-standard"};const T=n("div").withConfig({displayName:"DekAndCaption"})`
  ${b()}
  ${f("padding")}
  ${({isInverted:e,theme:t})=>e&&`\n      background: ${o(t,"colors.background.dark")};\n    `}
`,E=n.div.withConfig({displayName:"DekWrapper"})`
  grid-column: 1 / span 4;
  text-align: ${({contentAlign:e})=>e};

  @media (min-width: ${d.md}) {
    grid-column: 3 / span 8;
  }
`,N=n(c).withConfig({displayName:"Dek"})`
  ${({isInverted:e,theme:t})=>e&&`\n    color: ${o(t,"colors.consumption.lead.inverted.description")};\n    `}
  text-align: ${({contentAlign:e})=>e};
`;N.defaultProps={as:"p",bottomSpacing:4,colorToken:"colors.consumption.lead.standard.description",topSpacing:3,typeIdentity:"typography.definitions.consumptionEditorial.description-core"};const B=n(c).withConfig({displayName:"Figure"})`
  grid-column: 1 / span 4;
  text-align: ${({contentAlign:e})=>e};

  @media (min-width: ${d.md}) {
    grid-column: 1 / span 12;
  }
`;B.defaultProps={as:"figure",colorToken:"colors.consumption.lead.standard.description",topSpacing:2,typeIdentity:"typography.definitions.consumptionEditorial.description-embed"};const L=n.span.withConfig({displayName:"ContentDivider"})`
  display: block;
  margin-top: ${r(4)};
  border-bottom-width: 2px;
  border-bottom-style: solid;
  border-bottom-color: ${({theme:e})=>o(e,"colors.consumption.lead.standard.accent")};
  width: 100px;
  ${({contentAlign:e})=>"center"===e&&`margin: ${r(4)} auto 0`}
`,I=n.div.withConfig({displayName:"ContributorImage"})`
  display: block;
  margin-top: ${r(4)};
  border-radius: 50%;
  min-width: 60px;
  max-width: 66px;
  min-height: 60px;
  max-height: 66px;
  overflow: hidden;
  ${({contentAlign:e})=>"center"===e&&`margin: ${r(4)} auto 0`}
`,A=n.div.withConfig({displayName:"Accreditation"})`
  ${({contentAlign:e})=>"center"===e?`margin: ${r(2)} auto`:`margin: ${r(2)} 0`}
`,P=n.time.withConfig({displayName:"PublishDate"})`
  ${({theme:e})=>a(e,"typography.definitions.globalEditorial.accreditation-core")}
  display: block;
  margin: ${r(1)} 0 ${r(4)};
  text-align: ${({contentAlign:e})=>e};
  color: ${({theme:e})=>o(e,"colors.consumption.lead.inverted.context-tertiary")};
`,R=n(h.WithMargins).withConfig({displayName:"ContentGrid"})`
  > ${y} {
    grid-column: 1 / span 4;
    margin-bottom: 4.5rem;
    @media (min-width: ${d.md}) {
      grid-column: ${({contentAlign:e})=>"left"===e?"1 / span 10":"2 / span 10"};
    }
  }
`,D=n(C).withConfig({displayName:"TextOverlayLogoImage"})`
  grid-column: 1 / span 4;

  img {
    max-width: 100%;
    height: 100px;
    vertical-align: bottom;
  }
`,M=n(p).withConfig({displayName:"TextOverlayLogoLink"})`
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: ${r(2)};

  @media (max-width: ${d.md}) {
    padding-right: calc(1 * ${r(3)});
    padding-left: calc(1 * ${r(3)});
  }
`,O=n.div.withConfig({displayName:"TextOverlayLogo"})`
  margin: auto;
  margin-top: 1.5rem;
`;e.exports={TextOverlayLogo:O,TextOverlayLogoLink:M,TextOverlayLogoImage:D,TextOverlayWrapper:v,ImageWrapper:w,Image:S,ContentAlign:k,Content:x,Hed:$,DekAndCaption:T,DekWrapper:E,Dek:N,Figure:B,ContentDivider:L,ContributorImage:I,Accreditation:A,PublishDate:P,ContentGrid:R}},505:function(e,t,i){const{default:n}=i(3),{BaseText:o}=i(10),{calculateSpacing:a,getColorToken:r,getTypographyStyles:l}=i(4),s=n.div.withConfig({displayName:"ScoreBoxWrapper"})`
  position: relative;
  width: 130px;
`,d=n.div.withConfig({displayName:"ScoreCircle"})`
  position: relative;
  margin-bottom: ${a(2)};
  border: 7px solid
    ${({isBest:e})=>e?({theme:e})=>"inverted"===e.palette?r("colors.consumption.lead.inverted.accent"):r("colors.consumption.lead.standard.accent"):r("colors.consumption.lead.standard.context-signature")};
  border-radius: 50%;
  padding-bottom: ${a(2)};
  width: 134px;
  height: 134px;
  text-align: center;
  color: ${({isBest:e})=>e?({theme:e})=>"inverted"===e.palette?r("colors.consumption.lead.inverted.accent"):r("colors.consumption.lead.standard.accent"):r("colors.consumption.lead.standard.context-signature")};
`,c=n(o).withConfig({displayName:"Rating"})`
  display: block;
  position: absolute;
  top: 50%;
  transform: translateY(-50%);
  width: 100%;
  color: ${({isBest:e})=>e?({theme:e})=>"inverted"===e.palette?r("colors.consumption.lead.inverted.accent"):r("colors.consumption.lead.standard.accent"):r("colors.consumption.lead.standard.context-signature")};
  ${l("typography.definitions.globalEditorial.numerical-large")};
`,p=n(o).withConfig({displayName:"BestNewMusicText"})`
  ${l("typography.definitions.globalEditorial.context-primary")};
  width: 134px;
  text-align: center;
  color: ${({isBest:e})=>e?({theme:e})=>"inverted"===e.palette?r("colors.consumption.lead.inverted.accent"):r("colors.consumption.lead.standard.accent"):r("colors.consumption.lead.standard.context-signature")};
`,g=n.svg.withConfig({displayName:"SvgStyle"})`
  margin-bottom: ${a(2,"px")};
  margin-left: ${a(3,"px")};
  width: 86px;
  height: 26px;
`,u=n.div.withConfig({displayName:"SvgWrapper"})`
  svg {
    fill: ${({isBest:e})=>e?({theme:e})=>"inverted"===e.palette?r("colors.consumption.lead.inverted.accent"):r("colors.consumption.lead.standard.accent"):r("colors.consumption.lead.standard.context-signature")};
  }
  line-height: 0em;
`;e.exports={BestNewMusicText:p,Rating:c,ScoreBoxWrapper:s,ScoreCircle:d,SvgStyle:g,SvgWrapper:u}},532:function(e,t,i){const n=i(1),o=i(0),a=i(8),{BlockquoteEmbedWrapper:r,BlockquoteEmbedContent:l,BlockquoteEmbedFooter:s,BlockquoteEmbedCite:d}=i(304),{trackComponent:c}=i(5),p=({attributes:e,children:t,citeUrl:i,className:n,dangerousAttribution:p,hasParagraphMargin:g,hasSmallMargins:u,hasTopBorder:m,shouldUseBodyColor:h})=>(o.useEffect(()=>{c("BlockquoteEmbed")},[]),o.createElement(r,Object.assign({},e,{cite:i,hasTopBorder:m,hasSmallMargins:u,className:a(n,"blockquote-embed")}),o.createElement(l,{hasParagraphMargin:g,shouldUseBodyColor:h,className:"blockquote-embed__content"},t),p&&o.createElement(s,null,o.createElement(d,{dangerouslySetInnerHTML:{__html:p}}))));p.propTypes={attributes:n.object,children:n.oneOfType([n.arrayOf(n.node),n.node]).isRequired,citeUrl:n.string,className:n.string,dangerousAttribution:n.string,hasParagraphMargin:n.bool,hasSmallMargins:n.bool,hasTopBorder:n.bool,shouldUseBodyColor:n.bool},p.defaultProps={hasSmallMargins:!1,hasTopBorder:!0,shouldUseBodyColor:!1},p.displayName="BlockquoteEmbed",e.exports=p},715:function(e,t,i){const n=i(8),o=i(1),a=i(0),{default:r}=i(162),{trackComponent:l}=i(5),{processLinks:s,processCeros:d,processTiktok:c,processSidebarHeadings:p}=i(268),g=i(470),u=i(260),{BodyWrapper:m}=i(169),h=new r({a:s,blockquote:({props:e})=>({type:g,props:e}),ceros:d,h2:p,tiktok:c,"inline-embed":u}),y=({body:e,className:t,children:i,shouldDisableMaxWidth:o,shouldEnableDataJourneyHook:r=!0})=>{a.useEffect(()=>{l("Body")},[]);const s={className:n("body",t),shouldDisableMaxWidth:o};return r&&(s["data-journey-hook"]="client-content"),a.createElement(m,Object.assign({},s,{"data-testid":"BodyWrapper"}),i||h.convert(e))};y.propTypes={body:o.array,children:o.node,className:o.string,shouldDisableMaxWidth:o.bool,shouldEnableDataJourneyHook:o.bool},y.defaultProps={body:["div"],shouldDisableMaxWidth:!1},e.exports=y},80:function(e,t,i){const{asConfiguredComponent:n}=i(9),o=i(150);e.exports=n(o,"BasePage")}});