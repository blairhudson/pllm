(set-option :timeout 10000)

(set-logic QF_BV)
(declare-const x0 (_ BitVec 8))
(declare-const x1 (_ BitVec 8))
(declare-const r (_ BitVec 8))
(assert (not (= (bvsub (bvsub x1 r) (bvsub x0 r)) (bvsub x1 x0))))
(check-sat)
