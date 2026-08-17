#!/bin/bash
# Spin-orbit spillage on Fir (Alliance/DRAC). One material per submission.
#
# Submit from the REPOSITORY ROOT -- every path below is relative to it:
#
#   sbatch --export=ALL,MAT=Bi2Se3 slurm/fir_spillage_fx.sbatch
#   sbatch --export=ALL,MAT=Bi2Te3 slurm/fir_spillage_fx.sbatch
#
# Runs the non-SOC then the SOC SCF, then the overlap post-processor.
# Both SCFs must use the identical cell, cutoffs and k-mesh (nosym/noinv) so the
# G-vector lists line up -- that is what compute_spillage.py relies on.

#SBATCH --account=def-YOURACCOUNT   # set to your Alliance/DRAC allocation
#SBATCH --job-name=spillfx
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=48
#SBATCH --mem-per-cpu=2G
#SBATCH --output=%x-%j.out

set -euo pipefail

: "${MAT:?set MAT=Bi2Se3 or MAT=Bi2Te3 via --export=ALL,MAT=...}"

module --force purge
module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 quantumespresso/7.5

command -v pw.x >/dev/null || { echo "pw.x not on PATH -- fix the module load"; exit 1; }

NPOOL=24                                 # 216 k-points / 24 pools = 9 per pool, 2 procs each

for TAG in nosoc soc; do
    IN="inputs/${MAT}.scf.${TAG}.fx.in"
    OUT="${TAG}_fx_${MAT}.out"
    echo "=== ${MAT} ${TAG} : $(date) ==="
    srun pw.x -npool ${NPOOL} -in "${IN}" > "${OUT}"
    grep -q "JOB DONE" "${OUT}" || { echo "pw.x did not finish for ${MAT} ${TAG}"; exit 1; }
    grep -q "convergence has been achieved" "${OUT}" || { echo "SCF did not converge for ${MAT} ${TAG}"; exit 1; }
done

# The post-processor reads wfc*.hdf5 out of the two .save directories.
NS_SAVE="out_nosoc_fx_${MAT}/${MAT}_nosoc_fx.save"
SOC_SAVE="out_soc_fx_${MAT}/${MAT}_soc_fx.save"
ls "${NS_SAVE}"/wfc1.hdf5 "${SOC_SAVE}"/wfc1.hdf5 >/dev/null || {
    echo "No wfc*.hdf5 in the .save dirs -- cannot compute the overlap."; exit 1; }

module load python/3.11
source "$HOME/venv-spillage/bin/activate"
SPILLAGE_NELEC=78 python compute_spillage.py "${NS_SAVE}" "${SOC_SAVE}" \
    | tee "spillage_fx_${MAT}_out.txt"

echo "=== done $(date) ==="
