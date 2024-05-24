use nova_snark::{
  provider::{Bn256EngineKZG, GrumpkinEngine},
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

use flate2::{write::ZlibEncoder, Compression};

type E1 = Bn256EngineKZG;
type E2 = GrumpkinEngine;
type EE1 = nova_snark::provider::hyperkzg::EvaluationEngine<E1>;
type EE2 = nova_snark::provider::ipa_pc::EvaluationEngine<E2>;
type S1 = nova_snark::spartan::snark::RelaxedR1CSSNARK<E1, EE1>; // non-preprocessing SNARK
type S2 = nova_snark::spartan::snark::RelaxedR1CSSNARK<E2, EE2>; // non-preprocessing SNARK

/*
a Fibonacci circuit
for example
4,7
7,11
11,18
18,29
29,47
*/
#[derive(Clone, Debug)]
struct FibonacciInstance<G: Group> {
num1: G::Scalar,
num2: G::Scalar,
}

impl<G: Group> FibonacciInstance<G> {
  fn new(a: u64, b: u64) -> Self { 
    let num1 = G::Scalar::from(a);
    let num2 = G::Scalar::from(b);

    Self{
      num1,
      num2,
    }
  }
}


#[derive(Clone, Debug)]
struct FibonacciCircuit<G: Group> {
seq: Vec<FibonacciInstance<G>>,
}

impl<G: Group> FibonacciCircuit<G> {
  fn new(start1: u64, start2: u64, step: usize) -> Self {
    let mut seq = Vec::new();
    let mut num1 = start1;
    let mut num2 = start2;
    for _ in 0..step{
      seq.push(FibonacciInstance::new(num1, num2));
      num2 = num1 + num2;
      num1 = num2 - num1;
    }
    Self { seq }
  }

  fn recursive_fibo(from: [u64;2], step:usize, routine: usize) -> Vec<Self>{
    let mut out = Vec::new();
    let mut parameter1 = Vec::new();
    let mut parameter2 = Vec::new();
    let mut num1 = from[0];
    let mut num2 = from[1];
    parameter1.push(from[0]);
    parameter2.push(from[1]);
    for i in 0..step*routine - 2{
      num2 = num1 + num2;
      num1 = num2 - num1;
      if (i+1)%step == 0{
        parameter1.push(num1);
        parameter2.push(num2);
      }
    }

    for (i, j) in parameter1.iter().zip(parameter2.iter()){
      out.push(FibonacciCircuit::new(*i, *j, step));
    }
    out
  }
}


impl<G: Group> StepCircuit<G::Scalar> for FibonacciCircuit<G> {
  fn arity(&self) -> usize {
    1
  }

  fn synthesize<CS: ConstraintSystem<G::Scalar>>(
    &self,
    cs: &mut CS,
    z_in: &[AllocatedNum<G::Scalar>],
  ) -> Result<Vec<AllocatedNum<G::Scalar>>, SynthesisError> {
    let mut out: Result<Vec<AllocatedNum<G::Scalar>>, SynthesisError> =
    Err(SynthesisError::AssignmentMissing);

    for i in 0..self.seq.len()-1{
      let num1 = AllocatedNum::alloc(cs.namespace(|| format!("num1_{}", i)), || {
        Ok(self.seq[i].num1)
      })?;
      let num2 = AllocatedNum::alloc(cs.namespace(|| format!("num2_{}", i)), || {
        Ok(self.seq[i].num2)
      })?;
      let num1_next = AllocatedNum::alloc(cs.namespace(|| format!("num1_next_{}", i)), || {
        Ok(self.seq[i+1].num1)
      })?;
      let num2_next = AllocatedNum::alloc(cs.namespace(|| format!("num2_next_{}", i)), || {
        Ok(self.seq[i+1].num2)
      })?;

      cs.enforce(
        || format!("num1 + num2 = num2_next"),
        |lc| lc + num1.get_variable() + num2.get_variable(),
        |lc| lc + CS::one(),
        |lc| lc + num2_next.get_variable(),
      );
      cs.enforce(
        || format!("num2 = num1_next"),
        |lc| lc + num2.get_variable(),
        |lc| lc + CS::one(),
        |lc| lc + num1_next.get_variable(),
      );
      
      if i == self.seq.len() - 2 {
        out = Ok(vec![num2_next]);
      }
    }
    out
}
    
}

fn main() {
  println!("Nova-based VDF with Fibonacci");
  println!("=========================================================");
  type Cruve = halo2curves::bn256::G1;

  let step = 5;
  let start1 = 4;
  let start2 = 7;
  let circuit_primary = FibonacciCircuit::new(start1, start2, step);
  let circuit_secondary = TrivialCircuit::default();
  let routine = 10;

  // produce public parameters
  println!("Producing public parameters...");
  let pp = PublicParams::<
    E1,
    E2,
    FibonacciCircuit<<E1 as Engine>::GE>,
    TrivialCircuit<<E2 as Engine>::Scalar>,
  >::setup(
    &circuit_primary,
    &circuit_secondary,
    &*S1::ck_floor(),
    &*S2::ck_floor(),
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
  
  type C1 = FibonacciCircuit<<E1 as Engine>::GE>;
  type C2 = TrivialCircuit<<E2 as Engine>::Scalar>;

  let circuits = FibonacciCircuit::<Cruve>::recursive_fibo([start1,start2], step, routine);

  // produce a recursive SNARK
  println!("Generating a RecursiveSNARK...");
  let mut recursive_snark: RecursiveSNARK<E1, E2, C1, C2> =
    RecursiveSNARK::<E1, E2, C1, C2>::new(
      &pp,
      &circuits[0],
      &circuit_secondary,
      &[<E1 as Engine>::Scalar::zero()],
      &[<E2 as Engine>::Scalar::zero()]
    )
    .unwrap();
  
    for circuit_primary in circuits.iter() {
      let res = recursive_snark.prove_step(&pp, circuit_primary, &circuit_secondary);
      assert!(res.is_ok());
    }
    
    // verify the recursive SNARK
    println!("Verifying a RecursiveSNARK...");
    let res = recursive_snark.verify(
      &pp,
      routine,
      &[<E1 as Engine>::Scalar::zero()],
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
  
    let compressed_snark_encoded = encoder.finish().unwrap();
    println!(
      "CompressedSNARK::len {:?} bytes",
      compressed_snark_encoded.len()
    );

    // verify the compressed SNARK
    println!("Verifying a CompressedSNARK...");
    let res = compressed_snark.verify(
      &vk,
      routine,
      &[<E1 as Engine>::Scalar::zero()],
      &[<E2 as Engine>::Scalar::zero()],
    );
    println!(
      "CompressedSNARK::verify: {:?}",res.is_ok()
    );
    assert!(res.is_ok());
    println!("=========================================================");
}