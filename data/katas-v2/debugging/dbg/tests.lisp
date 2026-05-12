;;; tests.lisp — hand-crafted, inspired by Rust's dbg!
;;; instruction: Write `dbg` — print "form = value", return value.
;;; category: debugging | technique: tap | complexity: basic
;;; quality_score: 0.95

(
 ((dbg (compute-x))
  . (LET ((IT (COMPUTE-X)))
      (FORMAT *TRACE-OUTPUT* "dbg: ~S = ~S~%" '(COMPUTE-X) IT)
      IT))

 ((dbg (+ a b))
  . (LET ((IT (+ A B)))
      (FORMAT *TRACE-OUTPUT* "dbg: ~S = ~S~%" '(+ A B) IT)
      IT))

 ((dbg user)
  . (LET ((IT USER))
      (FORMAT *TRACE-OUTPUT* "dbg: ~S = ~S~%" 'USER IT)
      IT))
)
