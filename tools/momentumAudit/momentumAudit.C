// Read-only boundary momentum export. No solver, turbulence correction or
// evaluation of the saved codedFixedValue boundary conditions is performed.
#include "fvCFD.H"
#include "IOdictionary.H"
#include "OFstream.H"
#include "IFstream.H"
#include "IStringStream.H"
#include "OStringStream.H"

using namespace Foam;

template<class Type>
dictionary frozenDictionary
(
    const word& name, const word& fieldClass, const fvMesh& mesh,
    const label internalSize, const bool volume
)
{
    IOdictionary saved
    (
        IOobject(name,mesh.time().timeName(),mesh,IOobject::MUST_READ,IOobject::NO_WRITE,false),
        fieldClass
    );
    dictionary dict(saved);
    autoPtr<Field<Type>> internal;
    dictionary& boundary=dict.subDict("boundaryField");
    forAll(mesh.boundary(),patchi)
    {
        const fvPatch& patch=mesh.boundary()[patchi];
        if (patch.type()!="patch" && patch.type()!="wall")
        {
            FatalErrorInFunction << "Only reconstructed serial patch/wall meshes are supported: "
                << patch.name() << ' ' << patch.type() << exit(FatalError);
        }
        dictionary& entry=boundary.subDict(patch.name());
        const word originalType=entry.get<word>("type");
        if (!entry.found("value"))
        {
            if (!volume || originalType!="zeroGradient")
            {
                FatalErrorInFunction << "Missing stored value for " << name << '/' << patch.name()
                    << " (only volume zeroGradient can be reconstructed from owner cells)"
                    << exit(FatalError);
            }
            if (!internal.valid()) internal.reset(new Field<Type>("internalField",dict,internalSize));
            Field<Type> value(patch.size());
            forAll(value,i) value[i]=internal()[patch.faceCells()[i]];
            OStringStream text;
            text.precision(17);
            value.writeEntry("value",text);
            IStringStream input(text.str());
            const dictionary values(input);
            entry.merge(values);
        }
        // Retain saved values, but never instantiate/execute the original BC.
        entry.set("type",word("fixedValue"));
    }
    if (internal.valid())
    {
        // Field's dictionary constructor transfers a binary compound token.
        // Restore a fresh entry before GeometricField reads it a second time.
        OStringStream text;
        text.precision(17);
        internal().writeEntry("internalField",text);
        IStringStream input(text.str());
        const dictionary values(input);
        dict.merge(values);
    }
    return dict;
}

void csvVector(Ostream& os,const vector& v)
{
    os << ',' << v.x() << ',' << v.y() << ',' << v.z();
}

int main(int argc,char* argv[])
{
    argList::addNote("Frozen-field export for steady incompressible kOmegaSST momentum audits.");
    argList::noParallel();
    argList::addOption("audit-out","directory","Output directory (outside the CFD case)");
    argList::addOption("nu","scalar","Molecular kinematic viscosity, m2/s");
    argList::addOption("cord-width","scalar","Cord diameter, m");
    argList::addOption("cord-cd","scalar","Cord Cd");
    argList::addOption("cord-samples","integer","Samples per cord segment");
    #include "setRootCase.H"
    #include "createTime.H"
    #include "createMesh.H"
    const scalar nu=args.getOrDefault<scalar>("nu",1.46e-5);
    const scalar width=args.getOrDefault<scalar>("cord-width",0.0032);
    const scalar cd=args.getOrDefault<scalar>("cord-cd",1.2);
    const label samples=args.getOrDefault<label>("cord-samples",64);
    if (nu<=0 || width<=0 || cd<=0 || samples<1)
        FatalErrorInFunction << "Invalid physical parameters" << exit(FatalError);
    const fileName output=args.get<fileName>("audit-out");
    mkDir(output);

    const dictionary ud=frozenDictionary<vector>("U","volVectorField",mesh,mesh.nCells(),true);
    const dictionary pd=frozenDictionary<scalar>("p","volScalarField",mesh,mesh.nCells(),true);
    const dictionary nd=frozenDictionary<scalar>("nut","volScalarField",mesh,mesh.nCells(),true);
    const dictionary fd=frozenDictionary<scalar>("phi","surfaceScalarField",mesh,mesh.nInternalFaces(),false);
    volVectorField U(IOobject("U",runTime.timeName(),mesh,IOobject::NO_READ,IOobject::NO_WRITE),mesh,ud);
    volScalarField p(IOobject("p",runTime.timeName(),mesh,IOobject::NO_READ,IOobject::NO_WRITE),mesh,pd);
    volScalarField nut(IOobject("nut",runTime.timeName(),mesh,IOobject::NO_READ,IOobject::NO_WRITE),mesh,nd);
    surfaceScalarField phi(IOobject("phi",runTime.timeName(),mesh,IOobject::NO_READ,IOobject::NO_WRITE),mesh,fd);
    if (U.dimensions()!=dimensionSet(0,1,-1,0,0,0,0)
        || p.dimensions()!=dimensionSet(0,2,-2,0,0,0,0)
        || nut.dimensions()!=dimensionSet(0,2,-1,0,0,0,0)
        || phi.dimensions()!=dimensionSet(0,3,-1,0,0,0,0))
        FatalErrorInFunction << "Expected incompressible U, kinematic p/nut and volume phi" << exit(FatalError);
    const volTensorField gradient(fvc::grad(U));
    const volSymmTensorField tau
    (
        (dimensionedScalar("auditNu",dimViscosity,nu)+nut)*dev(twoSymm(gradient))
    );
    OFstream faces(output/"boundary.csv");
    faces.precision(17);
    faces << "patch,face,cx,cy,cz,sx,sy,sz,p_kin,Ux,Uy,Uz,phi,tauSfx,tauSfy,tauSfz\n";
    forAll(mesh.boundary(),patchi)
    {
        const fvPatch& patch=mesh.boundary()[patchi];
        const vectorField& centres=mesh.Cf().boundaryField()[patchi];
        const vectorField& areas=mesh.Sf().boundaryField()[patchi];
        forAll(patch,i)
        {
            const vector shear=tau.boundaryField()[patchi][i] & areas[i];
            faces << patch.name() << ',' << patch.start()+i;
            csvVector(faces,centres[i]); csvVector(faces,areas[i]);
            faces << ',' << p.boundaryField()[patchi][i];
            csvVector(faces,U.boundaryField()[patchi][i]);
            faces << ',' << phi.boundaryField()[patchi][i];
            csvVector(faces,shear); faces << '\n';
        }
    }
    // The bounded convection scheme subtracts U*div(phi). Report this term
    // separately instead of interpreting a continuity defect as fabric force.
    const volScalarField divergence(fvc::div(phi));
    scalar l1=0,net=0;
    vector boundedForce(Zero),boundedMoment(Zero);
    forAll(mesh.V(),celli)
    {
        const scalar flux=divergence[celli]*mesh.V()[celli];
        const vector force=flux*U[celli];
        l1+=mag(flux); net+=flux;
        boundedForce+=force;
        boundedMoment+=mesh.C()[celli]^force;
    }
    OFstream volume(output/"volume.csv");
    volume.precision(17);
    volume << "quantity,x,y,z\ncontinuity_l1," << l1 << ",0,0\ncontinuity_net," << net << ",0,0\n";
    volume << "bounded_force_kin"; csvVector(volume,boundedForce); volume << '\n';
    volume << "bounded_moment_kin"; csvVector(volume,boundedMoment); volume << '\n';

    IFstream cordInput(mesh.time().path()/mesh.time().constant()/"cords");
    if (!cordInput.good())
        FatalErrorInFunction << "Missing constant/cords" << exit(FatalError);
    const vectorField ends(cordInput);
    if (ends.empty() || ends.size()%2)
        FatalErrorInFunction << "Invalid cord endpoint pairs" << exit(FatalError);
    OFstream cords(output/"cord_mapping.csv");
    cords.precision(17);
    cords << "sample,cell,cx,cy,cz,x,y,z,rx_kin,ry_kin,rz_kin\n";
    for (label j=0;j<ends.size()/2;++j)
    {
        const vector delta=ends[2*j+1]-ends[2*j];
        const scalar length=mag(delta);
        if (length<=SMALL) FatalErrorInFunction << "Zero length cord" << exit(FatalError);
        const vector tangent=delta/length;
        const scalar weight=0.5*cd*width*length/samples;
        for (label i=0;i<samples;++i)
        {
            const label k=j*samples+i;
            const vector point=ends[2*j]+((i+0.5)/samples)*delta;
            const label cell=mesh.findCell(point);
            if (cell<0) FatalErrorInFunction << "Cord sample outside fluid: " << k << exit(FatalError);
            const vector transverse=U[cell]-(U[cell]&tangent)*tangent;
            const vector reaction=weight*mag(transverse)*transverse;
            cords << k << ',' << cell;
            csvVector(cords,mesh.C()[cell]); csvVector(cords,point); csvVector(cords,reaction);
            cords << '\n';
        }
    }
    Info<< "Frozen fields exported to " << output << nl;
    return 0;
}
