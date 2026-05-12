;;; tests.lisp — hand-crafted, Alexandria-style
;;; instruction: Write `if-let` — like IF but binds a caller-named var
;;; to the test result so BOTH branches can use it.
;;; category: capture-management | technique: name-capture | complexity: basic
;;; quality_score: 0.95

(
 ((if-let (user (find-user id)) (greet user) (signal :no-user))
  . (LET ((USER (FIND-USER ID))) (IF USER (GREET USER) (SIGNAL :NO-USER))))

 ((if-let (m (match regex s)) (extract m) :nomatch)
  . (LET ((M (MATCH REGEX S))) (IF M (EXTRACT M) :NOMATCH)))

 ((if-let (n (parse-integer s :junk-allowed t)) (* n n) 0)
  . (LET ((N (PARSE-INTEGER S :JUNK-ALLOWED T))) (IF N (* N N) 0)))
)
