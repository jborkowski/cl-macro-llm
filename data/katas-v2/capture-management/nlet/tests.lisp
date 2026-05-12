;;; tests.lisp — hand-crafted from Let Over Lambda ch. 5
;;; instruction: Write `nlet` — Scheme-style named let. The loop name is
;;; captured into a LABELS form so the body can recurse.
;;; category: capture-management | technique: name-capture | complexity: intermediate
;;; quality_score: 0.95

(
 ((nlet loop ((n 10) (acc 0))
    (if (zerop n) acc (loop (1- n) (+ acc n))))
  . (LABELS ((LOOP (N ACC)
               (IF (ZEROP N) ACC (LOOP (1- N) (+ ACC N)))))
      (LOOP 10 0)))

 ((nlet count-down ((i 5))
    (when (plusp i) (print i) (count-down (1- i))))
  . (LABELS ((COUNT-DOWN (I)
               (WHEN (PLUSP I) (PRINT I) (COUNT-DOWN (1- I)))))
      (COUNT-DOWN 5)))

 ((nlet walk ((node tree))
    (when node (walk (left node)) (visit node) (walk (right node))))
  . (LABELS ((WALK (NODE)
               (WHEN NODE (WALK (LEFT NODE)) (VISIT NODE) (WALK (RIGHT NODE)))))
      (WALK TREE)))
)
