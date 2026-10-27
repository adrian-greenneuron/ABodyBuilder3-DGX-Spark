import importlib.util
from pathlib import Path
import unittest

import numpy as np
import openmm as mm
from openmm import app,unit

spec=importlib.util.spec_from_file_location('rescue',Path(__file__).resolve().parents[1]/'scripts/heavy_atom_rescue.py')
rescue=importlib.util.module_from_spec(spec);spec.loader.exec_module(rescue)


class LocalRescueTests(unittest.TestCase):
    def test_bond_shell_has_no_unbonded_or_hydrogen_atoms(self):
        top=app.Topology();r=top.addResidue('GLY',top.addChain())
        atoms=[top.addAtom(str(i),app.element.hydrogen if i==3 else app.element.carbon,r) for i in range(5)]
        for a,b in [(0,1),(1,2),(1,3)]:top.addBond(atoms[a],atoms[b])
        self.assertEqual(rescue.bond_shell(top,[0],1),{0,1})
        self.assertEqual(rescue.bond_shell(top,[0],2),{0,1,2})

    def test_chirality_wall_penalises_inversion_with_correct_force(self):
        top=app.Topology();r=top.addResidue('ALA',top.addChain());c=top.addAtom('CA',app.element.carbon,r)
        for name,el in [('N',app.element.nitrogen),('C',app.element.carbon),('CB',app.element.carbon),('HA',app.element.hydrogen)]:top.addBond(c,top.addAtom(name,el,r))
        xyz=np.array([[0,0,0],[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]],float)*.1
        system=mm.System()
        for _ in range(5):system.addParticle(1)
        system.addForce(rescue.orientation_guard(top,xyz,biological_only=True))
        integrator=mm.VerletIntegrator(0);context=mm.Context(system,integrator,mm.Platform.getPlatformByName('Reference'))
        try:
            context.setPositions(xyz);self.assertEqual(context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole),0)
            xyz[3]*=-1;context.setPositions(xyz);state=context.getState(getEnergy=True,getForces=True)
            self.assertGreater(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole),0)
            force=state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole/unit.nanometer)[3,1]
            values=[]
            for sign in [-1,1]:
                moved=xyz.copy();moved[3,1]+=sign*1e-7;context.setPositions(moved)
                values.append(context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
            self.assertAlmostEqual(force/(-(values[1]-values[0])/2e-7),1,places=6)
        finally:del context,integrator

    def test_neighbourhood_never_crosses_a_chain_boundary(self):
        top=app.Topology();chains=[top.addChain('H'),top.addChain('L')];atoms=[]
        for chain in chains:
            for _ in range(3):
                r=top.addResidue('GLY',chain);atoms.append(top.addAtom('CA',app.element.carbon,r));top.addAtom('HA',app.element.hydrogen,r)
        self.assertEqual(rescue.neighbourhood(top,[atoms[2].index],1),{atoms[1].index,atoms[2].index})
        self.assertEqual(rescue.neighbourhood(top,[atoms[2].index],0),{atoms[2].index})

    def test_chiral_reference_changes_sign_after_reflection(self):
        top=app.Topology();r=top.addResidue('ALA',top.addChain());c=top.addAtom('CA',app.element.carbon,r)
        for name,el in [('N',app.element.nitrogen),('C',app.element.carbon),('CB',app.element.carbon),('HA',app.element.hydrogen)]:top.addBond(c,top.addAtom(name,el,r))
        xyz=np.array([[0,0,0],[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]],float)
        a=next(iter(rescue.chiral_volumes(top,xyz).values()));xyz[:,0]*=-1
        b=next(iter(rescue.chiral_volumes(top,xyz).values()));self.assertLess(a*b,0)

    def test_local_minimisation_fixes_a_bond_without_moving_frozen_atoms(self):
        top=app.Topology();chain=top.addChain();r1=top.addResidue('GLY',chain);a=top.addAtom('CA',app.element.carbon,r1)
        r2=top.addResidue('GLY',chain);b=top.addAtom('CA',app.element.carbon,r2);top.addBond(a,b)
        # The source can be the massless-heavy system used by addHydrogens.
        system=mm.System();system.addParticle(0);system.addParticle(0)
        bond=mm.HarmonicBondForce();bond.addBond(0,1,.15,100000);system.addForce(bond)
        pos=np.array([[0,0,0],[.25,0,0]])*unit.nanometer
        new,info=rescue.repair(system,top,pos,[b.index],radius=0,restraint_k=1000)
        xyz=np.asarray(new.value_in_unit(unit.nanometer));np.testing.assert_array_equal(xyz[0],[0,0,0])
        self.assertLess(abs(np.linalg.norm(xyz[1]-xyz[0])-.15),.002)
        self.assertTrue(info['fixed_atoms_unchanged'])


if __name__=='__main__':unittest.main()
