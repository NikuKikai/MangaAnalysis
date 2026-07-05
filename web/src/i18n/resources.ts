export const resources = {
  en: {
    translation: {
      loading: {
        boot: "Initializing",
        webgpu: "Initializing WebGPU",
        model: "Loading Model",
        ready: "Ready",
        error: "Error",
      },
      panel: {
        sections: {
          panels: "Panels",
          preprocess: "Preprocess",
          saliency: "Saliency",
          history: "History",
          selector: "Selector",
          panelBoxesNote: "Detect panel boxes when an image opens and draw them on the overlay.",
        },
        language: {
          label: "Language",
          en: "English",
          ja: "日本語",
        },
        buttons: {
          openImage: "Open Image",
          clickMode: "Click Mode",
          boxMode: "Box Mode",
          nextStep: "Next Step",
          reset: "Reset",
          information: "Information",
          languageMenu: "Open Language Menu",
          save: "Save Visualization",
          expand: "Expand Panel",
          collapse: "Collapse Panel",
          saliencyOnlyStrategy: "Saliency",
          panelGuidedStrategy: "Panel Guided",
          togglePanels: "Toggle Panel Boxes",
          togglePreprocess: "Toggle Preprocess View",
          toggleSaliency: "Toggle Saliency View",
          toggleHistory: "Toggle History View",
          toggleSelector: "Toggle Candidate Overlay",
        },
        info: {
          dialogLabel: "Information",
          author: "Author: NikuKikai",
          contact: "Contact: nikukikai@gmail.com",
          socialLinks: "Social links",
          github: "GitHub",
          x: "X",
        },
        sliders: {
          blur: {
            label: "Blur",
            tooltip:
              "Maximum blur radius used outside\nthe clear fovea.\nCurrent value: {{value}} px.\nLarger values make peripheral regions blurrier\nand harder to read.\nSmaller values keep more detail.",
          },
          fovea: {
            label: "Fovea",
            tooltip:
              "Radius of the sharp center area\naround the fixation point.\nFormula: image height ({{imageHeight}})\nx ratio ({{ratio}})\n= {{pixels}}.\nLarger values keep a wider area unblurred.\nSmaller values make blur start closer\nto the fixation point.",
          },
          box: {
            label: "Box",
            tooltip:
              "Half-size of the default square ROI used to start each step.\nClick mode uses this size directly.\nBox mode only chooses the initial ROI for the current step.\nFormula: image height ({{imageHeight}})\nx ratio ({{ratio}})\n= half-size {{halfSize}}.\nFull ROI size is {{fullSize}}.\nLarger values inspect a wider area\nwith lower local detail.\nSmaller values focus on a tighter\nlocal neighborhood.",
          },
          sigma: {
            label: "Sigma",
            tooltip:
              "Spread of each fixation deposit written\ninto the history map.\nFormula: image height ({{imageHeight}})\nx ratio ({{ratio}})\n= sigma {{pixels}}.\nLarger values spread inhibition across\na broader area.\nSmaller values keep inhibition concentrated\nnear the fixation point.",
          },
          hist: {
            label: "Hist",
            tooltip:
              "Strength of history-based suppression\nin candidate scoring.\nCurrent value: {{value}}.\nThe score multiplier is\nexp(-alpha x historyValue).\nLarger values penalize revisits more aggressively.\nSmaller values make history matter less.",
          },
          decay: {
            label: "Decay",
            tooltip:
              "How quickly old history fades\nwhen a new fixation is added.\nCurrent value: {{value}}.\nNew history is accumulated as\noldValue x decay + gaussian.\nLarger values preserve history longer.\nSmaller values erase old inhibition faster.",
          },
          dist: {
            label: "Dist",
            tooltip:
              "Spread of the distance preference\naround the current fixation.\nFormula: image height ({{imageHeight}})\nx ratio ({{ratio}})\n= sigma {{pixels}}.\nLarger values weaken the preference\nfor nearby candidates.\nSmaller values bias the next fixation\nmore strongly toward nearby locations.",
          },
          thresh: {
            label: "Thresh",
            tooltip:
              "Post-filter threshold applied\nto final candidate scores.\nCurrent value: {{value}}.\nCandidates with finalScore below this value\nare discarded.\nLarger values keep only stronger peaks.\nSmaller values allow weaker candidates\nto survive.",
          },
          nms: {
            label: "Nms",
            tooltip:
              "Non-maximum suppression radius\nin model-space cells.\nCurrent estimate:\nround(image height ({{imageHeight}})\nx ratio ({{ratio}})\nx model size 512\n/ ROI size ({{roiSource}}: {{roiSize}}))\n= {{cells}} cells.\nLarger values merge nearby peaks\nmore aggressively.\nSmaller values keep more local maxima.",
          },
          topK: {
            label: "TopK",
            tooltip:
              "Maximum number of candidates kept\nafter GPU ranking and sorting.\nCurrent value: {{value}}.\nLarger values preserve more alternatives\nfor inspection.\nSmaller values keep only the strongest\nfew candidates.",
          },
        },
        roiSource: {
          current: "current ROI size",
          default: "default ROI size",
        },
        unknown: "unknown",
      },
    },
  },
  ja: {
    translation: {
      loading: {
        boot: "初期化中",
        webgpu: "WebGPU を初期化中",
        model: "モデルを読み込み中",
        ready: "準備完了",
        error: "エラー",
      },
      panel: {
        sections: {
          panels: "コマ",
          preprocess: "前処理",
          saliency: "顕著性",
          history: "履歴",
          selector: "選択",
          panelBoxesNote: "画像を開いたときにコマの box を検出し、オーバーレイ上に描画します。",
        },
        language: {
          label: "言語",
          en: "English",
          ja: "日本語",
        },
        buttons: {
          openImage: "画像を開く",
          clickMode: "クリックモード",
          boxMode: "ボックスモード",
          nextStep: "次へ",
          reset: "リセット",
          information: "情報",
          languageMenu: "言語メニューを開く",
          save: "可視化結果を保存",
          expand: "パネルを展開",
          collapse: "パネルを折りたたむ",
          saliencyOnlyStrategy: "Saliency",
          panelGuidedStrategy: "Panel Guided",
          togglePanels: "コマ box 表示の切り替え",
          togglePreprocess: "Preprocess 表示の切り替え",
          toggleSaliency: "Saliency 表示の切り替え",
          toggleHistory: "History 表示の切り替え",
          toggleSelector: "候補 overlay の切り替え",
        },
        info: {
          dialogLabel: "情報",
          author: "Author: NikuKikai",
          contact: "Contact: nikukikai@gmail.com",
          socialLinks: "ソーシャルリンク",
          github: "GitHub",
          x: "X",
        },
        sliders: {
          blur: {
            label: "Blur",
            tooltip:
              "注視点まわりのクリア領域の外側で使う\n最大ぼかし半径です。\n現在値: {{value}} px。\n大きいほど周辺部が強くぼけて読みにくくなり、\n小さいほど細部が残ります。",
          },
          fovea: {
            label: "Fovea",
            tooltip:
              "注視点のまわりで鮮明に保つ中心領域の半径です。\n計算式: 画像高さ ({{imageHeight}})\nx ratio ({{ratio}})\n= {{pixels}}。\n大きいほど非ぼかし領域が広がり、\n小さいほど注視点の近くからぼかしが始まります。",
          },
          box: {
            label: "Box",
            tooltip:
              "各ステップの開始時に使う既定の正方形 ROI の半サイズです。\nクリックモードではこのサイズを直接使います。\nボックスモードはそのステップだけの初期 ROI を選ぶためのものです。\n計算式: 画像高さ ({{imageHeight}})\nx ratio ({{ratio}})\n= 半サイズ {{halfSize}}。\nROI 全体サイズは {{fullSize}} です。\n大きいほど広い範囲を見ますが局所性は下がり、\n小さいほど狭い領域に集中します。",
          },
          sigma: {
            label: "Sigma",
            tooltip:
              "各注視点を history map に書き込むときの\n広がり幅です。\n計算式: 画像高さ ({{imageHeight}})\nx ratio ({{ratio}})\n= sigma {{pixels}}。\n大きいほど抑制が広い範囲に広がり、\n小さいほど注視点の近くに集中します。",
          },
          hist: {
            label: "Hist",
            tooltip:
              "候補スコア計算での history 抑制の強さです。\n現在値: {{value}}。\nスコア倍率は exp(-alpha x historyValue) です。\n大きいほど再訪を強く抑制し、\n小さいほど history の影響が弱くなります。",
          },
          decay: {
            label: "Decay",
            tooltip:
              "新しい注視点を加えたときに\n古い history がどれだけ残るかです。\n現在値: {{value}}。\n更新式は oldValue x decay + gaussian です。\n大きいほど history が長く残り、\n小さいほど古い抑制が早く消えます。",
          },
          dist: {
            label: "Dist",
            tooltip:
              "現在の注視点の近さを評価する\n距離優先の広がり幅です。\n計算式: 画像高さ ({{imageHeight}})\nx ratio ({{ratio}})\n= sigma {{pixels}}。\n大きいほど近距離優先が弱まり、\n小さいほど次の注視点が近場に寄りやすくなります。",
          },
          thresh: {
            label: "Thresh",
            tooltip:
              "最終候補スコアに対する\n後段フィルタのしきい値です。\n現在値: {{value}}。\nfinalScore がこの値未満の候補は捨てられます。\n大きいほど強いピークだけ残り、\n小さいほど弱い候補も残ります。",
          },
          nms: {
            label: "Nms",
            tooltip:
              "model 空間セル単位での\n非極大抑制半径です。\n現在の概算:\nround(画像高さ ({{imageHeight}})\nx ratio ({{ratio}})\nx model size 512\n/ ROI size ({{roiSource}}: {{roiSize}}))\n= {{cells}} cells。\n大きいほど近いピークを強くまとめ、\n小さいほど局所最大を多く残します。",
          },
          topK: {
            label: "TopK",
            tooltip:
              "GPU ランキング後に保持する\n候補数の上限です。\n現在値: {{value}}。\n大きいほど比較用の候補を多く残し、\n小さいほど上位の強い候補だけに絞ります。",
          },
        },
        roiSource: {
          current: "現在の ROI サイズ",
          default: "既定の ROI サイズ",
        },
        unknown: "不明",
      },
    },
  },
} as const;

export type AppLanguage = keyof typeof resources;
