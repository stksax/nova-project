// use nova_snark::{
//     provider::{PallasEngine, VestaEngine},
//     traits::{
//       circuit::{StepCircuit, TrivialCircuit},
//       snark::RelaxedR1CSSNARKTrait,
//       Engine, Group,
//     },
//     CompressedSNARK, PublicParams, RecursiveSNARK,
// };
// use core::marker::PhantomData;
// use bellpepper_core::{
//     boolean::AllocatedBit, num::AllocatedNum, ConstraintSystem, LinearCombination, SynthesisError,
//   };

// type E1 = PallasEngine;
// type E2 = VestaEngine;
// type EE1 = nova_snark::provider::hyperkzg::EvaluationEngine<E1>;
// type EE2 = nova_snark::provider::ipa_pc::EvaluationEngine<E2>;
// type S1 = nova_snark::spartan::snark::RelaxedR1CSSNARK<E1, EE1>; // non-preprocessing SNARK
// type S2 = nova_snark::spartan::snark::RelaxedR1CSSNARK<E2, EE2>; // non-preprocessing SNARK


// #[derive(Clone, Debug)]
// struct AndInstance<G: Group> {
//   a: G::Scalar,
// }

// impl<G: Group> AndInstance<G> {
//     // produces an AND instance
//     fn new() -> Self {
//       let a: u64 = 12;
//       Self {
//         a,
//         _p: PhantomData,
//       }
//     }
//   }
  
//     #[derive(Clone, Debug)]
//     struct AndCircuit<G: Group> {
//         batch: Vec<AndInstance<G>>,
//     }
    
//     impl<G: Group> AndCircuit<G> {
//         // produces a batch of AND instances
//         fn new(num_ops_per_step: usize) -> Self {
//         let mut batch = Vec::new();
//         for _ in 0..num_ops_per_step {
//             batch.push(AndInstance::new());
//         }
//         Self { batch }
//         }
//     }


// impl<G: Group> StepCircuit<G::Scalar> for AndCircuit<G> {
//     fn arity(&self) -> usize {
//       1
//     }
  
//     fn synthesize<CS: ConstraintSystem<G::Scalar>>(
//       &self,
//       cs: &mut CS,
//       z_in: &[AllocatedNum<G::Scalar>],
//     ) -> Result<Vec<AllocatedNum<G::Scalar>>, SynthesisError> {
//         let a = AllocatedNum::alloc(cs, value)
//     }
//   }

use std::ops::{Add, Mul};
use pasta_curves::{arithmetic::CurveExt, group::{cofactor::CofactorCurveAffine, ff::PrimeField}, pallas};

fn main(){
  let g = pallas::Affine::generator();
  let s1 = pallas::Scalar::from_u128(123);
  let p1 = pallas::Affine::mul(g, s1);
  let (x1,y1,z1) = p1.jacobian_coordinates();

  let s2 = pallas::Scalar::from_u128(456);
  let p2 = pallas::Affine::mul(g, s2);
  let (x2,y2,z2) = p2.jacobian_coordinates();

  let a = x1 * z2.square();
  let b = x2 * z1.square() - a;
  let c = y1 * z2.square() * z2;
  let d = y2 * z1.square() * z1 - c;
  let z3 = z1 * z2 * b;
  let x3 = d.square() - b.square() * (b + a.double());
  let y3 = d * (a * b.square() - x3) - c * b.square() * b;

  let p3 = pallas::Point::new_jacobian(x3, y3, z3).unwrap();
  let p4 = pallas::Point::add(p1, p2);
  let result = pallas::Point::eq(&p3, &p4);
  assert_eq!(result,true);
}