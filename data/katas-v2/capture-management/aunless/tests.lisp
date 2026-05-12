;;; tests.lisp — hand-crafted from On Lisp ch. 14
;;; instruction: Write `aunless` — like UNLESS but the test result is
;;; captured as IT so the body can refer to it (e.g. for diagnostics).
;;; category: capture-management | technique: anaphoric | complexity: basic
;;; quality_score: 0.95

(
 ((aunless (find-user id) (signal-missing it))
  . (LET ((IT (FIND-USER ID))) (UNLESS IT (SIGNAL-MISSING IT))))

 ((aunless (cached-value key) (compute-and-cache key))
  . (LET ((IT (CACHED-VALUE KEY))) (UNLESS IT (COMPUTE-AND-CACHE KEY))))

 ((aunless (verify-token tok) (log :rejected it) (raise 'invalid-token))
  . (LET ((IT (VERIFY-TOKEN TOK)))
      (UNLESS IT (LOG :REJECTED IT) (RAISE 'INVALID-TOKEN))))
)
