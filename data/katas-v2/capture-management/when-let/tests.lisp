;;; tests.lisp — hand-crafted, Alexandria-style
;;; instruction: Write `when-let` — binds a caller-named variable to a
;;; test result and runs the body when non-nil. Caller-named, not
;;; anaphoric: the symbol is captured into the binding.
;;; category: capture-management | technique: name-capture | complexity: basic
;;; quality_score: 0.95

(
 ((when-let (user (find-user 42)) (greet user))
  . (LET ((USER (FIND-USER 42))) (WHEN USER (GREET USER))))

 ((when-let (record (lookup :id 7)) (print record) (touch record))
  . (LET ((RECORD (LOOKUP :ID 7))) (WHEN RECORD (PRINT RECORD) (TOUCH RECORD))))

 ((when-let (n (parse-integer s :junk-allowed t)) (* n 2))
  . (LET ((N (PARSE-INTEGER S :JUNK-ALLOWED T))) (WHEN N (* N 2))))
)
