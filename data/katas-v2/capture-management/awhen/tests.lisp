;;; tests.lisp — hand-crafted from On Lisp ch. 14
;;; instruction: Write a Common Lisp macro `awhen` (anaphoric when) that
;;; captures the test result in the symbol IT so the body can refer to it.
;;; category: capture-management | technique: anaphoric | complexity: basic
;;; quality_score: 0.95

(
 ((awhen (lookup-user 42) (greet it))
  . (LET ((IT (LOOKUP-USER 42))) (WHEN IT (GREET IT))))

 ((awhen (assoc :name record) (cdr it))
  . (LET ((IT (ASSOC :NAME RECORD))) (WHEN IT (CDR IT))))

 ((awhen (find-error log) (report it) (recover it))
  . (LET ((IT (FIND-ERROR LOG))) (WHEN IT (REPORT IT) (RECOVER IT))))
)
