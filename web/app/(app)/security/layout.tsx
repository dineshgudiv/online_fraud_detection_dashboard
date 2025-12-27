import Link from "next/link";

const tabs = [
  { label: "Overview", href: "/security" },
  { label: "PKI", href: "/security/pki" },
  { label: "Certificates", href: "/security/certificates" },
  { label: "HSM", href: "/security/hsm" }
];

export default function SecurityLayout({
  children
}: {
  children: React.ReactNode;
}) {
  return (
    <section className="section">
      <div className="security-header">
        <div>
          <h3>Security Center</h3>
          <p className="muted">
            PKI posture, certificate lifecycle (Venafi), and HSM-backed key
            custody.
          </p>
        </div>
        <div className="status-pill">Trust: monitoring</div>
      </div>

      <div className="subnav">
        {tabs.map((tab) => (
          <Link key={tab.href} href={tab.href} className="subnav-link">
            {tab.label}
          </Link>
        ))}
      </div>

      {children}
    </section>
  );
}
