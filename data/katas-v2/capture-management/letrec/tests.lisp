;;; tests.lisp — hand-crafted, Scheme/LoL flavor
;;; instruction: Write `letrec` — Scheme-style mutually-recursive
;;; function bindings via labels.
;;; category: capture-management | technique: name-capture | complexity: intermediate
;;; quality_score: 0.9

(
 ((letrec ((even? (n) (if (zerop n) t (odd? (1- n))))
           (odd?  (n) (if (zerop n) nil (even? (1- n)))))
    (even? 10))
  . (LABELS ((EVEN? (N) (IF (ZEROP N) T (ODD? (1- N))))
             (ODD?  (N) (IF (ZEROP N) NIL (EVEN? (1- N)))))
      (EVEN? 10)))

 ((letrec ((ping (x) (if (zerop x) :done (pong (1- x))))
           (pong (x) (if (zerop x) :done (ping (1- x)))))
    (ping 5))
  . (LABELS ((PING (X) (IF (ZEROP X) :DONE (PONG (1- X))))
             (PONG (X) (IF (ZEROP X) :DONE (PING (1- X)))))
      (PING 5)))
)
