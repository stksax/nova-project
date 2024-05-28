#![allow(dead_code, unused_imports)]
use std::ops::{Add, Mul};
use ff::PrimeField;
use halo2curves::{group::cofactor::CofactorCurveAffine, CurveExt};
use nova_snark::{
  provider::{PallasEngine, VestaEngine},
  traits::{
    circuit::{StepCircuit, TrivialCircuit},
    snark::RelaxedR1CSSNARKTrait,
    Engine, Group,
  },
  CompressedSNARK, PublicParams, RecursiveSNARK,
};
use bellpepper_core::{
  num::AllocatedNum, ConstraintSystem, SynthesisError
};
use bincode;
use flate2::{write::ZlibEncoder, Compression};
use pasta_curves::{pallas, vesta};
use nova_snark::provider::ipa_pc::EvaluationEngine;
use nova_snark::spartan::ppsnark::RelaxedR1CSSNARK;

type EE<E> = EvaluationEngine<E>;
type SPrime<E> = RelaxedR1CSSNARK<E, EE<E>>;
type E1 = VestaEngine;
type E2 = PallasEngine;
type S1 = nova_snark::spartan::snark::RelaxedR1CSSNARK<E1, EE<E1>>; // non-preprocessing SNARK
type S2 = nova_snark::spartan::snark::RelaxedR1CSSNARK<E2, EE<E2>>; // non-preprocessing SNARK


#[derive(Clone, Debug)]
struct EccAddInstance<G: Group> {
  x1: G::Scalar,
  y1: G::Scalar,
  z1: G::Scalar,
  x2: G::Scalar,
  y2: G::Scalar,
  z2: G::Scalar,
  negative_num: [G::Scalar; 5]
}

fn new_instance (point1: pallas::Point, point2: pallas::Point) -> EccAddInstance<vesta::Point>{
    let (x1, y1, z1) = point1.jacobian_coordinates();
    let (x2, y2, z2) = point2.jacobian_coordinates();
  
    let a = x1 * z2.square();
    let b = x2 * z1.square() - a;
    let c = y1 * z2.square() * z2;
    let d = y2 * z1.square() * z1 - c;
    let z3 = z1 * z2 * b;
    let x3 = d.square() - b.square() * (b + a.double());
    let y3 = d * (a * b.square() - x3) - c * b.square() * b;
  
    let neg_a = a.neg();
    let neg_c = c.neg();
    let neg_b2_mul_b_plus_2a = x3 - d.square();
    let neg_x3 = x3.neg();
    let neg_c_3b = (c * b.square() * b).neg();

    //make sure jacobian add is correct
    let point3 = pallas::Point::new_jacobian(x3, y3, z3).unwrap();
    let eqcheck = pallas::Point::eq(&point3, &point1.add(point2));
    assert_eq!(eqcheck,true);

    let negative_num = [neg_a, neg_c, neg_b2_mul_b_plus_2a, neg_x3, neg_c_3b];
    
    EccAddInstance{
      x1,y1,z1,x2,y2,z2,negative_num
    }
}

#[derive(Clone, Debug)]
struct EccAddCircuit<G: Group> {
  seq: Vec<EccAddInstance<G>>,
}

fn new_circuit (point_vec: Vec<pallas::Point>) -> EccAddCircuit<vesta::Point>{
  let mut out = Vec::new();
  let mut point1 = point_vec[0];
  let mut point2 = point_vec[1];
  let (mut x1, mut y1, mut z1) = point1.jacobian_coordinates();
  let (mut x2, mut y2, mut z2) = point2.jacobian_coordinates();
  
  let mut a = x1 * z2.square();
  let mut b = x2 * z1.square() - a;
  let mut c = y1 * z2.square() * z2;
  let mut d = y2 * z1.square() * z1 - c;
  let mut z3 = z1 * z2 * b;
  let mut x3 = d.square() - b.square() * (b + a.double());
  let mut y3 = d * (a * b.square() - x3) - c * b.square() * b;
  let mut point3 = pallas::Point::new_jacobian(x3, y3, z3).unwrap();
  let mut input = new_instance (point1, point2);
  out.push(input);
  for i in 1..point_vec.len()-1{
    point1 = point3;
    point2 = point_vec[i+1];
    (x1, y1, z1) = point1.jacobian_coordinates();
    (x2, y2, z2) = point2.jacobian_coordinates();
    
    a = x1 * z2.square();
    b = x2 * z1.square() - a;
    c = y1 * z2.square() * z2;
    d = y2 * z1.square() * z1 - c;
    z3 = z1 * z2 * b;
    x3 = d.square() - b.square() * (b + a.double());
    y3 = d * (a * b.square() - x3) - c * b.square() * b;
    point3 = pallas::Point::new_jacobian(x3, y3, z3).unwrap();
    input = new_instance(point1, point2);
    out.push(input);
  }
  
  EccAddCircuit { seq: out }
}

impl<G: Group> StepCircuit<G::Scalar> for EccAddCircuit<G> {
  //three output represent jacobian_coordinates of point
  fn arity(&self) -> usize {
    3
  }

  fn synthesize<CS: ConstraintSystem<G::Scalar>>(
    &self,
    cs: &mut CS,
    z_in: &[AllocatedNum<G::Scalar>],
  ) -> Result<Vec<AllocatedNum<G::Scalar>>, SynthesisError> {
    let mut out: Result<Vec<AllocatedNum<G::Scalar>>, SynthesisError> =
    Err(SynthesisError::AssignmentMissing);

    for i in 0..self.seq.len(){
      //point1 jacobian_coordinates
      let x1 = AllocatedNum::alloc(
        cs.namespace(|| format!("x1")), || {
        Ok(self.seq[i].x1)
      })?;
      let y1 = AllocatedNum::alloc(
        cs.namespace(|| format!("y1")), || {
        Ok(self.seq[i].y1)
      })?;
      let z1 = AllocatedNum::alloc(
        cs.namespace(|| format!("z1")), || {
        Ok(self.seq[i].z1)
      })?;

      //point2 jacobian_coordinates
      let x2 = AllocatedNum::alloc(
        cs.namespace(|| format!("x2")), || {
        Ok(self.seq[i].x2)
      })?;
      let y2 = AllocatedNum::alloc(
        cs.namespace(|| format!("y2")), || {
        Ok(self.seq[i].y2)
      })?;
      let z2 = AllocatedNum::alloc(
        cs.namespace(|| format!("z2")), || {
        Ok(self.seq[i].z2)
      })?;
      
      //allocate negative number
      let neg_a = AllocatedNum::alloc(
        cs.namespace(|| format!("-a")), || {
        Ok(self.seq[i].negative_num[0])
      })?;
      let neg_c = AllocatedNum::alloc(
        cs.namespace(|| format!("-c")), || {
        Ok(self.seq[i].negative_num[1])
      })?;
      let neg_b2_mul_b_plus_2a = AllocatedNum::alloc(
        cs.namespace(|| format!("-(b^2 * (b+2a))")), || {
        Ok(self.seq[i].negative_num[2])
      })?;
      let neg_x3 = AllocatedNum::alloc(
        cs.namespace(|| format!("-x3")), || {
        Ok(self.seq[i].negative_num[3])
      })?;
      let neg_c_3b = AllocatedNum::alloc(
        cs.namespace(|| format!("-c*3b")), || {
        Ok(self.seq[i].negative_num[4])
      })?;

      //calculate negative number(verify with precompute parameter) and other parameter
      let z2_square = z2.square(cs.namespace(|| format!("z^2"))).unwrap();
      let a = AllocatedNum::mul(&x1, cs.namespace(|| format!("x * z^2")), &z2_square).unwrap();
      cs.enforce(
        || format!("-a = -(x1 * (z2^2))"),
        |lc| lc + a.get_variable() + neg_a.get_variable(),
        |lc| lc + CS::one(),
        |lc| lc ,
      );

      let z1_square = z1.square(cs.namespace(|| format!("z1^2"))).unwrap();
      let b = AllocatedNum::mul(&x2, cs.namespace(|| format!("x * z1^2")), &z1_square).unwrap();
      let b = AllocatedNum::add(&b, cs.namespace(|| format!("(x * z1^2) - a")), &neg_a).unwrap();
   
      let c = AllocatedNum::mul(&y1, cs.namespace(|| format!("non")), &z2_square).unwrap();
      let c = AllocatedNum::mul(&c, cs.namespace(|| format!("non")), &z2).unwrap();
      cs.enforce(
        || format!("-c = -(y1 * (z2^3))"),
        |lc| lc + c.get_variable() + neg_c.get_variable(),
        |lc| lc + CS::one(),
        |lc| lc ,
      );

      let d = AllocatedNum::mul(&y2, cs.namespace(|| format!("non")), &z1_square).unwrap();
      let d = AllocatedNum::mul(&d, cs.namespace(|| format!("non")), &z1).unwrap();
      let d = AllocatedNum::add(&d, cs.namespace(|| format!("non")), &neg_c).unwrap();

      let z3 = AllocatedNum::mul(&z1, cs.namespace(|| format!("z1 * z2")), &z2).unwrap(); 
      let z3 = AllocatedNum::mul(&z3, cs.namespace(|| format!("z1 * z2 * b")), &b).unwrap();
      
      let double_a = AllocatedNum::add(&a, cs.namespace(|| format!("non")), &a).unwrap();
      let b_plus_2a = AllocatedNum::add(&b, cs.namespace(|| format!("non")), &double_a).unwrap();
      let b_square = b.square(cs.namespace(|| format!("non"))).unwrap();
      let b2_mul_b_plus_2a = AllocatedNum::mul(&b_square, cs.namespace(|| format!("non")), &b_plus_2a).unwrap();
      cs.enforce(
        || format!("-b^2 * (b + 2a)"),
        |lc| lc + neg_b2_mul_b_plus_2a.get_variable()+ b2_mul_b_plus_2a.get_variable(),
        |lc| lc + CS::one(),
        |lc| lc ,
      );
      
      let d_square = d.square(cs.namespace(|| format!("non"))).unwrap();
      let x3 = AllocatedNum::add(&d_square, cs.namespace(|| format!("non")), &neg_b2_mul_b_plus_2a).unwrap();
      cs.enforce(
        || format!("-x3"),
        |lc| lc + x3.get_variable() + neg_x3.get_variable(),
        |lc| lc + CS::one(),
        |lc| lc ,
      );
      
      let c_b_cube  = AllocatedNum::mul(&c, cs.namespace(|| format!("non")), &b_square).unwrap();
      let c_b_cube = AllocatedNum::mul(&c_b_cube, cs.namespace(|| format!("non")), &b).unwrap();
      cs.enforce(
        || format!("-c * (b^3)"),
        |lc| lc + c_b_cube.get_variable() + neg_c_3b.get_variable(),
        |lc| lc + CS::one(),
        |lc| lc ,
      );
      
      let y3 = AllocatedNum::mul(&a, cs.namespace(|| format!("non")), &b_square).unwrap();
      let y3 = AllocatedNum::add(&y3, cs.namespace(|| format!("non")), &neg_x3).unwrap();
      let y3 = AllocatedNum::mul(&d, cs.namespace(|| format!("non")), &y3).unwrap();
      let y3 = AllocatedNum::add(&y3,cs.namespace(|| format!("non")), &neg_c_3b).unwrap();
      
      if i < self.seq.len() -1{
        //allocate new point coordinates
        let next_x1 = AllocatedNum::alloc(
          cs.namespace(|| format!("next x1")), || {
          Ok(self.seq[i+1].x1)
        })?;
        let next_y1 = AllocatedNum::alloc(
          cs.namespace(|| format!("next y1")), || {
          Ok(self.seq[i+1].y1)
        })?;
        let next_z1 = AllocatedNum::alloc(
          cs.namespace(|| format!("next z1")), || {
          Ok(self.seq[i+1].z1)
        })?;

        //ensure the new point is the same as we calculate
        cs.enforce(
          || format!("point1_i + point2_i = point1_i+1"),
          |lc| lc + x3.get_variable(),
          |lc| lc + CS::one(),
          |lc| lc + next_x1.get_variable(),
        );

        cs.enforce(
          || format!("point1_i + point2_i = point1_i+1"),
          |lc| lc + y3.get_variable(),
          |lc| lc + CS::one(),
          |lc| lc + next_y1.get_variable(),
        );

        cs.enforce(
          || format!("point1_i + point2_i = point1_i+1"),
          |lc| lc + z3.get_variable(),
          |lc| lc + CS::one(),
          |lc| lc + next_z1.get_variable(),
        );
      } else {
        out = Ok(vec![x3,y3,z3]);
      }
    }
    out
  }
}

#[cfg(test)]
mod tests{
  use super::*;
  #[test]
fn ecc_point_add() {
  println!("Nova-based VDF with EccAdd");
  println!("=========================================================");

  //we have a list of point need to add
  let g = pallas::Affine::generator();
  let scalar = pallas::Scalar::from_u128(2222);
  let initial_point = pallas::Affine::mul(g, scalar);
  let scalar1 = pallas::Scalar::from_u128(123);
  let point1 = pallas::Affine::mul(g, scalar1);
  let scalar2 = pallas::Scalar::from_u128(456);
  let point2 = pallas::Affine::mul(g, scalar2);
  let scalar3 = pallas::Scalar::from_u128(321);
  let point3 = pallas::Affine::mul(g, scalar3);
  let scalar4 = pallas::Scalar::from_u128(654);
  let point4 = pallas::Affine::mul(g, scalar4);

  //we calculate the sum of point here, and verify if it is the same as result we calculate in the end
  let sum_num = 2222+123+456+321+654;
  let scalar_num = pallas::Scalar::from_u128(sum_num);
  let point_num = pallas::Affine::mul(g, scalar_num);

  let point_vec = vec![initial_point,point1,point2,point3,point4];
  let circuit = new_circuit(point_vec);

  let circuit_secondary = TrivialCircuit::default();

  //produce public parameters
  println!("Producing public parameters...");
  let pp = PublicParams::<
    E1,
    E2,
    EccAddCircuit<<E1 as Engine>::GE>,
    TrivialCircuit<<E2 as Engine>::Scalar>,
  >::setup(
    &circuit,
    &circuit_secondary,
    &*SPrime::<E1>::ck_floor(),
    &*SPrime::<E2>::ck_floor(),
  )
  .unwrap();
  
  println!(
    "Number of constraints per step (primary circuit): {}",
    pp.num_constraints().0
  );
  println!(
    "Number of constraints per step (secondary circuit): {}",
    pp.num_constraints().1
  );

  println!(
    "Number of variables per step (primary circuit): {}",
    pp.num_variables().0
  );
  println!(
    "Number of variables per step (secondary circuit): {}",
    pp.num_variables().1
  );
  
  type C1 = EccAddCircuit<<E1 as Engine>::GE>;
  type C2 = TrivialCircuit<<E2 as Engine>::Scalar>;

  // produce a recursive SNARK
  println!("Generating a RecursiveSNARK...");
  let mut recursive_snark: RecursiveSNARK<E1, E2, C1, C2> =
    RecursiveSNARK::<E1, E2, C1, C2>::new(
      &pp,
      &circuit,
      &circuit_secondary,
      &[<E1 as Engine>::Scalar::zero();3],
      &[<E2 as Engine>::Scalar::zero()]
    )
    .unwrap();
  
 
  let res = recursive_snark.prove_step(&pp, &circuit, &circuit_secondary);
  assert!(res.is_ok());
    
  // verify the recursive SNARK
  println!("Verifying a RecursiveSNARK...");
  let res = recursive_snark.verify(
    &pp,
    1,
    &[<E1 as Engine>::Scalar::zero();3],
    &[<E2 as Engine>::Scalar::zero()],
  );
  println!("RecursiveSNARK::verify: {:?}", res.is_ok(),);
  assert!(res.is_ok());

  // produce a compressed SNARK
  println!("Generating a CompressedSNARK using Spartan with HyperKZG...");
  let (pk, vk) = CompressedSNARK::<_, _, _, _, S1, S2>::setup(&pp).unwrap();

  let res = CompressedSNARK::<_, _, _, _, S1, S2>::prove(&pp, &pk, &recursive_snark);
  println!(
    "CompressedSNARK::prove: {:?}",res.is_ok()
  );
  assert!(res.is_ok());
  let compressed_snark = res.unwrap();

  let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
  bincode::serialize_into(&mut encoder, &compressed_snark).unwrap();

  let compressed_snark_encoded = encoder.finish().unwrap();
  println!(
    "CompressedSNARK::len {:?} bytes",
    compressed_snark_encoded.len()
  );

  // verify the compressed SNARK
  println!("Verifying a CompressedSNARK...");
  let res = compressed_snark.verify(
    &vk,
    1,
    &[<E1 as Engine>::Scalar::zero();3],
    &[<E2 as Engine>::Scalar::zero()],
  );
  println!(
    "CompressedSNARK::verify: {:?}",res.is_ok()
  );
  assert!(res.is_ok());

  //check result_point equal to point_sum
  let (out,_) = RecursiveSNARK::outputs(&recursive_snark);
  let result_point = pallas::Point::new_jacobian(out[0],out[1],out[2]).unwrap();
  let point_eq_check = pallas::Point::eq(&result_point, &point_num);
  assert_eq!(point_eq_check,true);
  println!("point from the circuit output is as we expect");
  println!("=========================================================");
}
}