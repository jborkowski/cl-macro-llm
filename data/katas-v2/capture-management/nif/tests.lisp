;;; tests.lisp — hand-crafted from On Lisp ch. 14
;;; instruction: Write `nif` — three-way numeric branch: positive,
;;; zero, negative. The value is captured as `it` for all branches.
;;; category: capture-management | technique: anaphoric | complexity: intermediate
;;; quality_score: 0.9

(
 ((nif (compute-balance) (deposit it) :zero (withdraw it))
  . (LET ((IT (COMPUTE-BALANCE)))
      (IF (PLUSP IT)
          (DEPOSIT IT)
          (IF (ZEROP IT)
              :ZERO
              (WITHDRAW IT)))))

 ((nif (signum delta) :up :flat :down)
  . (LET ((IT (SIGNUM DELTA)))
      (IF (PLUSP IT)
          :UP
          (IF (ZEROP IT)
              :FLAT
              :DOWN))))
)
