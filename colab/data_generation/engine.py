"""Deterministic security-scenario engine for Astra v4 distillation data.

The teacher API (Gemini) is throttled to roughly a 7% acceptance rate, so asking it
for labels cannot reach a 10k-row corpus. This engine inverts the problem: each
scenario *constructs* the log stream and the correct label from one random seed, so
the label is correct by construction instead of sampled from a busy model.

Everything is CPU-only, needs no network, and is reproducible from SEED.
"""

# --------------------------------------------------------------------------------------
# Entity pools. Drawn per record so the same scenario yields a different concrete
# stream each time - uniqueness comes from the entity space, not just the template.
# --------------------------------------------------------------------------------------

SVC_ACCOUNTS = ['svc-backup', 'svc-monitoring', 'svc-deploy', 'svc-backup-ad', 'svc-report',
                'svc-scan', 'svc-sync', 'svc-archive', 'svc-jenkins', 'svc-grafana']
HUMAN_USERS = ['a.okafor', 'm.lindqvist', 'r.deshmukh', 'j.park', 's.moreau', 't.eriksen',
               'l.fontaine', 'd.vasquez', 'k.novak', 'p.oyelaran', 'h.yamamoto', 'c.byrne']
CONTRACTORS = ['ext.consultant', 'vendor.sso', 'agency.contractor', 'partner.temp']
ROLES = ['web-01', 'web-02', 'db-primary', 'db-replica', 'file-share', 'jump-host', 'k8s-worker-03',
         'k8s-worker-07', 'auth-gateway', 'vpn-concentrator', 'print-floor', 'scanner-01',
         'build-runner', 'backup-nas', 'billing-app-02', 'proxy-egress', 'hypervisor-11',
         'legacy-rtu-07', 'pos-terminal-19', 'render-node-04', 'waf-edge-01', 'adfs-01']
IPS_INT = ['10.42.8.14', '10.42.8.77', '10.31.2.5', '10.60.4.19', '10.18.7.230', '10.9.1.12',
           '10.77.3.8', '10.55.6.41', '10.24.9.66', '10.36.5.17', '10.88.2.3', '10.12.4.88']
IPS_EXT = ['203.0.113.47', '198.51.100.22', '192.0.2.155', '203.0.113.201', '198.51.100.9',
           '192.0.2.77', '203.0.113.130', '198.51.100.164', '192.0.2.31', '203.0.113.88']
PORTS = [22, 3389, 445, 5985, 1433, 3306, 5432, 8080, 8443, 53, 123, 636]
PROCS = ['powershell.exe', 'cmd.exe', 'rundll32.exe', 'certutil.exe', 'psexesvc.exe',
         'wmic.exe', 'mshta.exe', 'net.exe', 'regsvr32.exe', 'svchost.exe', 'bash', 'curl']
PORTSHARE = ['\\\\fileshare\\finance', '\\\\fileshare\\legal', '\\\\nas01\\archive',
             '\\\\fileshare\\design', '\\\\nas01\\imaging']
LINUX_PATHS = ['/var/log/auth.log', '/home/%s/.ssh/authorized_keys', '/etc/passwd',
               '/var/backups/%s.sql', '/opt/app/config.yaml', '/tmp/.%s/cache.bin',
               '/usr/local/bin/deploy.sh', '/var/lib/%s/tokens.db']
CVE = ['CVE-2024-3094', 'CVE-2023-44487', 'CVE-2024-21762', 'CVE-2023-4966', 'CVE-2024-6387',
       'CVE-2022-42889', 'CVE-2023-38545', 'CVE-2024-27198']
PROD = ['payments-api', 'identity-broker', 'doc-portal', 'telemetry', 'billing-batch',
        'patient-portal', 'campus-sso', 'render-scheduler']
ENVIRONMENTS = [
    'a public cloud Kubernetes cluster',
    'a regional hospital network with legacy medical devices',
    'a university campus network during registration week',
    'a retail payment processing environment',
    'a small manufacturing plant with an industrial control segment',
    'a corporate datacentre with a corporate VPN',
    'a managed service provider moving between customer tenants',
    'a remote-first software company behind a zero-trust access proxy',
    'a freight and logistics operator with legacy telematics units',
    'a municipal water utility with remote monitoring stations',
    'a law firm with an externally hosted document management system',
    'a film studio with a distributed render farm and shared storage',
]

TIMINGS = [
    'during a normal working day',
    'in the small hours of the morning',
    'over a holiday weekend with only a skeleton crew on call',
    'in the fifteen minutes after a deployment finished',
    'during the weekly backup window',
    'shortly after a firewall rule change',
    'at the end of a billing cycle',
    'while a vendor was performing scheduled maintenance',
]

VARIANTS = [
    'a single analyst reviewing it with no other context',
    'two analysts who disagree about whether it matters',
    'a shift handover where the outgoing analyst was wrong earlier',
    'a weekend engineer with no access to the original request',
    'an auditor reconstructing the sequence from logs alone',
    'an incident review where the timeline matters more than the alerts',
    'a compliance question about what must be reported',
    'a follow-up investigation a week after the first alert',
    'a repeat offence by the same identity',
    'a first occurrence that has never been seen before',
    'a case where the evidence is incomplete and incomplete is the point',
    'a case where three separate tools each flagged something different',
    'a case where the benign explanation is proven by a change ticket',
    'a case where the benign explanation looks plausible but is wrong',
    'a case where severity depends entirely on which asset was hit',
    'a case where the same signature means different things on two hosts',
    'a case where the alert fired hours before anything actually happened',
    'a case where nothing happened yet and the question is what to do now',
    'a case where the user is a contractor with narrower expected access',
    'a case where service accounts and human accounts are being confused',
    'a case where a scheduled job is indistinguishable from an intruder',
    'a case where the only real signal is a missing event rather than an extra one',
    'a case where an attacker is deliberately staying inside the normal baseline',
    'a case where the correct answer is to escalate rather than to block',
]

# The analytical lens a record is written through. Scenarios choose their own content, so
# the lens is not required to match the stream; it widens the framing the model sees.
LENSES = [
    'a single quiet event that needs no explanation',
    'a burst of repeated failures from many sources',
    'an internal host reaching out for the first time',
    'a legitimate change that mimics an attack',
    'a scheduled job that looks wrong in isolation',
    'a service account whose scope grew unexpectedly',
    'a new device with no asset record',
    'a request that is odd for the host but normal for the domain',
    'a person who simply works unusual hours',
    'two unrelated alerts that turn out to be one incident',
    'a misconfigured sensor generating duplicate noise',
    'an expired certificate on a critical path',
    'a patch window producing a large legitimate footprint',
    'a third party with a risky but agreed integration',
    'a container pull that differs from the cluster baseline',
    'a noisy rule that fires on a real problem today',
    'an account whose second factor was removed',
    'a firewall rule widened while an incident was open',
    'a backup that grew several times overnight',
    'a certificate reissued shortly before expiry',
]

CITIES = ['Lisbon', 'Lagos', 'Osaka', 'Tallinn', 'Bogota', 'Helsinki', 'Jakarta', 'Dublin',
          'Cardiff', 'Porto', 'Bergen', 'Kaunas']


# --------------------------------------------------------------------------------------
# Log renderers. Six surface syntaxes so the model does not learn one vendor's dialect
# and then fail on another.
# --------------------------------------------------------------------------------------

def r_syslog(ev, rng):
    pri = {'critical': 38, 'error': 34, 'warning': 33, 'info': 30}[ev.get('sev', 'info')]
    mon = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][ev['m'] - 1]
    ts = '%s %2d %02d:%02d:%02d' % (mon, ev['d'], ev['h'], ev['mi'], ev['s'])
    head = '<%d>%s %s %s[%d]:' % (pri, ts, ev['host'], ev.get('app', 'app'), ev['pid'])
    return head + ' ' + ev['msg']


def r_auth(ev, rng):
    mon = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][ev['m'] - 1]
    ts = '%s %2d %02d:%02d:%02d' % (mon, ev['d'], ev['h'], ev['mi'], ev['s'])
    return '%s %s %s[%d]: %s' % (ts, ev['host'], ev.get('app', 'sshd'), ev['pid'], ev['msg'])


def r_cef(ev, rng):
    sev = {'critical': 9, 'error': 6, 'warning': 5, 'info': 2}[ev.get('sev', 'info')]
    # A fixed prefix on every line makes a multi-event CEF stream look like a loop to the
    # repetition gate, so vendor, product, version and signature are drawn from the event
    # rather than hardcoded. Real CEF exporters do vary these per rule set.
    vendor = ev.get('vendor') or rng.choice(
        ['Astra', 'Northwind', 'CorvidSec', 'Halcyon', 'BlueMesh', 'Vantage'])
    product = ev.get('product') or rng.choice(
        ['Labs Sensor', 'NDR', 'EDR', 'WireGuard', 'Sentinel', 'AuditTrail', 'ZettaCore'])
    version = ev.get('ver') or rng.choice(['1.0', '2.3', '3.1', '4.0.2'])
    ext = ' '.join('%s=%s' % (k, v) for k, v in ev.get('ext', {}).items())
    name = ev.get('name') or rng.choice(
        ['Event', 'Behaviour', 'Signal', 'Anomaly', 'Policy Hit', 'Access Event'])
    return 'CEF:%d|%s|%s|%s|%s|%d|%s|src=%s dst=%s %s' % (
        rng.choice([0, 1]), vendor, product, version, ev.get('sig', 'SIGNATURE'),
        sev, name, ev.get('src') or '-', ev.get('dst') or ev['host'], ext)


def r_json(ev, rng):
    body = {
        'ts': '2026-%02d-%02dT%02d:%02d:%02dZ' % (ev['m'], ev['d'], ev['h'], ev['mi'], ev['s']),
        'host': ev['host'], 'log': ev.get('app', 'app'), 'pid': ev['pid'],
        'event': ev['msg'],
    }
    for k in ('user', 'src', 'dst', 'port', 'result', 'action'):
        if ev.get(k):
            body[k] = ev[k]
    if ev.get('ext'):
        body['detail'] = ev['ext']
    import json as _j
    return _j.dumps(body, separators=(',', ':'))


def r_winsec(ev, rng):
    tail = ' '.join('%s: %s' % (k.capitalize(), v) for k, v in ev.get('ext', {}).items())
    # Only prefix an event ID when the event actually carries one, otherwise a non-Windows
    # event rendered in this dialect would read as a bogus security logon event.
    wid = ev.get('winid')
    head = '%d ' % wid if wid else ''
    return '%s%s %s %s | %s%s' % (head, ev['host'], ev.get('user', ev.get('src', 'system')),
                                  ev.get('log', ev.get('app', 'event')), ev['msg'],
                                  (' ' + tail) if tail else '')


def r_k8s(ev, rng):
    ts = '2026-%02d-%02dT%02d:%02d:%02d.%03dZ' % (ev['m'], ev['d'], ev['h'], ev['mi'],
                                                   ev['s'], rng.randrange(1000))
    return '%s %s %s: %s' % (ts, ev.get('app', 'app'), ev['host'], ev['msg'])


RENDERERS = [r_syslog, r_auth, r_cef, r_json, r_winsec, r_k8s]


# --------------------------------------------------------------------------------------
# Scenario library. Each builder returns the record with a label that is guaranteed to
# agree with the stream it just emitted, plus evidence naming the fields that decide it.
#
# A scenario returns: dict(cls, risk, early, evidence, analysis, action, conf, events)
# events are (importance, dict) tuples; importance drives which survive the char budget.
# --------------------------------------------------------------------------------------

SCENARIOS = {}


def scenario(category):
    def wrap(fn):
        SCENARIOS.setdefault(category, []).append(fn)
        return fn
    return wrap


def _ev(host, app, pid, msg, sev='info', imp=5, **kw):
    e = {'host': host, 'app': app, 'pid': pid, 'msg': msg, 'sev': sev, 'imp': imp}
    e.update(kw)
    return e


# ---------------------------------------------------------------- Normal behavior ----

@scenario('Normal behavior')
def backup_window(rng, env, lens, when):
    host, user, ip = rng.choice(ROLES), rng.choice(SVC_ACCOUNTS), rng.choice(IPS_INT)
    d = rng.randint(1, 28)
    n = rng.randrange(120, 900)
    ev = [
        _ev(host, 'cron', 2210, 'job backup-nightly started by %s per change CHG-%d' % (user, rng.randrange(4000, 4999)), imp=9),
        _ev(host, 'rsync', 2244, 'writing %d new objects to %s' % (n, PORTSHARE[0]), imp=8),
        _ev(host, 'rsync', 2244, 'checksum verified for %d files, 0 mismatches' % n, imp=7),
        _ev(host, 'cron', 2251, 'job backup-nightly finished rc=0 duration=412s', imp=9),
    ]
    return dict(cls='normal', risk='INFO', early=False, events=ev, conf=0.96,
                evidence=['job backup-nightly rc=0', 'checksum verified, 0 mismatches'],
                analysis='A scheduled backup job ran under its own service account, completed with rc=0 and verified checksums. Volume, timing and destination all match the documented baseline, so there is nothing anomalous here.',
                action='No incident response required. Retain the run log for the weekly backup attestation and confirm the schedule entry still matches the change record.')


@scenario('Normal behavior')
def deploy_pull(rng, env, lens, when):
    host, user, ip = rng.choice(ROLES), rng.choice(['ci-runner', 'devops', 'sre-bot']), rng.choice(IPS_INT)
    img = 'registry.internal/%s:%s' % (rng.choice(PROD), rng.choice(['v2.4.1', 'v2.4.2', 'sha-9f2c1a', 'v3.0.0']))
    ev = [
        _ev(host, 'containerd', 3301, 'pull request for %s from trusted registry' % img, imp=9),
        _ev(host, 'containerd', 3301, 'layer digest sha256:%s verified against signed manifest' % ''.join(rng.choice('0123456789abcdef') for _ in range(16)), imp=8),
        _ev(host, 'containerd', 3302, 'image %s admitted into namespace prod' % img.split('/')[-1], imp=8),
        _ev(host, 'kubelet', 3310, 'pod %s-%d started, image signature valid' % (rng.choice(PROD), rng.randrange(100, 999)), imp=7),
    ]
    return dict(cls='normal', risk='INFO', early=False, events=ev, conf=0.95,
                evidence=['image pulled from registry.internal', 'digest verified against signed manifest'],
                analysis='A container image came from the internal registry, its digest matched the signed manifest, and the resulting pod started without any capability changes or privilege escalation. This is the standard deployment path.',
                action='No action needed. The signed-manifest verification is the control that matters, so confirm it stays enabled for the internal registry in the next config review.')


@scenario('Normal behavior')
def scanner_sweep(rng, env, lens, when):
    src, sensor = rng.choice(IPS_INT), rng.choice(ROLES)
    ev = [
        _ev(sensor, 'nessus', 880, 'initiated TCP SYN probe from %s against %d hosts on subnet 10.42.8.0/24' % (src, rng.randrange(20, 400)), imp=9, src=src),
        _ev(sensor, 'nessus', 881, 'host discovery phase complete, %d live endpoints' % rng.randrange(20, 400), imp=8, src=src),
        _ev(sensor, 'nessus', 905, 'vulnerability scan phase started, %d plugins, authenticated' % rng.randrange(42, 200), imp=7, src=src),
    ]
    return dict(cls='normal', risk='INFO', early=False, events=ev, conf=0.94,
                evidence=['scan initiated by registered scanner host', 'authenticated scan with 42 plugins'],
                analysis='A high-volume fan-out of connection attempts is the registered vulnerability scanner doing subnet discovery. The source is the known scanner asset and the scan is authenticated, which is why no host is logging a failure.',
                action='No response needed. Confirm the scanner stays inside its agreed scan window, since the same signature from any other source would warrant investigation.')


@scenario('Normal behavior')
def failover(rng, env, lens, when):
    a, b = rng.sample(ROLES, 2)
    ev = [
        _ev(a, 'postgres', 4402, 'primary node lost heartbeat with standby for 3 intervals', sev='warning', imp=8),
        _ev(a, 'postgres', 4402, 'automatic failover triggered, promoting %s' % b, sev='warning', imp=9),
        _ev(b, 'postgres', 4410, 'promoted to primary, WAL replay caught up to %s' % rng.choice(['0/4A2F19C8', '0/71B3D0E5', '0/9C1E77A2']), imp=9),
        _ev(b, 'postgres', 4411, 'read-write service resumed, %d client connections re-established' % rng.randrange(40, 600), imp=7),
    ]
    return dict(cls='normal', risk='LOW', early=False, events=ev, conf=0.93,
                evidence=['automatic failover triggered by lost heartbeat', 'WAL replay caught up, clients re-established'],
                analysis='A database primary lost its heartbeat and the cluster promoted the standby automatically. Replay caught up and clients reconnected, which is the designed recovery path rather than a compromise.',
                action='Treat as availability work, not security. Confirm the failed primary is quarantined from the network before it is rebuilt, and review why the heartbeat was lost.')


@scenario('Normal behavior')
def afterhours_login(rng, env, lens, when):
    user, ip, host = rng.choice(HUMAN_USERS), rng.choice(IPS_EXT), rng.choice(ROLES)
    ev = [
        _ev(host, 'sshd', 5101, 'Accepted publickey for %s from %s port %d ssh2' % (user, ip, rng.randrange(40000, 60000)), imp=9, user=user, src=ip),
        _ev(host, 'sudo', 5102, '%s : TTY=pts/0 ; COMMAND=/usr/bin/less /var/log/audit/audit.log' % user, imp=8, user=user),
        _ev(host, 'sshd', 5103, 'session closed for user %s' % user, imp=7, user=user),
    ]
    return dict(cls='normal', risk='INFO', early=False, events=ev, conf=0.95,
                evidence=['publickey authentication succeeded', 'read-only log review under sudo'],
                analysis='An engineer authenticated with a public key outside working hours and then read an audit log with less. The account, method and activity are all consistent with routine log review, not credential misuse.',
                action='No escalation. If off-hours access is unexpected for this person, confirm it against the on-call rota rather than treating the login itself as suspicious.')


@scenario('Normal behavior')
def patch_window(rng, env, lens, when):
    host, n = rng.choice(ROLES), rng.randrange(18, 240)
    ev = [
        _ev(host, 'unattended-upgrades', 6610, 'patch window opened, maintenance CHG-%d authorised' % rng.randrange(5000, 5999), imp=9),
        _ev(host, 'dpkg', 6620, 'upgraded %d packages, %d restarts scheduled' % (n, rng.randrange(2, 30)), imp=8),
        _ev(host, 'systemd', 6630, 'service %s restarted, healthy after 3s' % rng.choice(PROD), imp=7),
        _ev(host, 'unattended-upgrades', 6640, 'patch window closed, %d reboots deferred to next cycle' % rng.randrange(0, 8), imp=8),
    ]
    return dict(cls='normal', risk='INFO', early=False, events=ev, conf=0.94,
                evidence=['authorised maintenance change record', 'package upgrades within window'],
                analysis='A broad change footprint is the scheduled patch window applying many package upgrades and restarting services. The activity is bounded by a change record and closed cleanly, which is what a maintenance burst looks like.',
                action='Confirm the change record is still open and that the deferred reboots get scheduled. Review the package list if any of the restarts touched an internet-facing service.')


@scenario('Normal behavior')
def cert_renewal(rng, env, lens, when):
    host = rng.choice(ROLES)
    ev = [
        _ev(host, 'acme-agent', 7100, 'certificate for %s.internal expiring in 14 days, renewal initiated' % rng.choice(PROD), imp=8),
        _ev(host, 'acme-agent', 7101, 'new key pair generated, RSA 2048, stored in KMS slot %d' % rng.randrange(2, 9), imp=9),
        _ev(host, 'acme-agent', 7102, 'certificate issued CN=%s.internal, valid 90 days' % rng.choice(PROD), imp=8),
        _ev(host, 'nginx', 7110, 'certificate reloaded without dropping connections', imp=7),
    ]
    return dict(cls='normal', risk='INFO', early=False, events=ev, conf=0.94,
                evidence=['renewal triggered 14 days before expiry', 'new key generated in KMS, not exported'],
                analysis='A certificate was renewed on the standard 14-day schedule, the replacement key was generated inside the KMS and the service reloaded without dropping connections. Nothing about the sequence is out of pattern.',
                action='No action. Verify the new key never left the KMS, and keep the 14-day renewal lead time so certificate expiry never becomes an incident.')


# ------------------------------------------------------- Suspicious behavior ----

@scenario('Suspicious behavior')
def password_spray(rng, env, lens, when):
    src = rng.choice(IPS_EXT)
    users = rng.sample(HUMAN_USERS, 5)
    ev = [_ev('auth-gateway', 'sshd', 5200, 'Failed password for %s from %s port %d ssh2' % (u, src, rng.randrange(40000, 60000)),
              sev='warning', imp=6 + (3 if i < 2 else 0), user=u, src=src) for i, u in enumerate(users)]
    ev.append(_ev('auth-gateway', 'sshd', 5299, 'authentication throttled for source %s after %d failures across %d distinct accounts' % (src, len(users) + 4, len(users)), sev='error', imp=10, src=src))
    return dict(cls='suspicious', risk='MEDIUM', early=True, events=ev, conf=0.89,
                evidence=['one source, %d distinct accounts, one attempt each' % len(users), 'gateway throttled the source after the spray'],
                analysis='A single external source tried one password against several different accounts. The pattern is too thin for any one account to lock out, which is the signature of spraying rather than brute force, and the gateway throttled it before a login succeeded.',
                action='Block the source at the perimeter, check whether any of the targeted accounts exist, and require MFA on externally reachable SSH so a password is no longer sufficient.')


@scenario('Suspicious behavior')
def new_device(rng, env, lens, when):
    ip, user, host = rng.choice(IPS_EXT), rng.choice(HUMAN_USERS), rng.choice(ROLES)
    ev = [
        _ev(host, 'edr', 8801, 'new device %s enrolled, no matching asset record or serial in CMDB' % ip, sev='warning', imp=9, src=ip),
        _ev(host, 'edr', 8802, 'device %s connected to %s over %s and presented a self-signed certificate' % (ip, host, rng.choice(['wireguard', 'rdp', 'https'])), sev='warning', imp=8, src=ip),
        _ev(host, 'edr', 8803, 'first connection from this device in %d days' % rng.randrange(90, 700), sev='warning', imp=9, src=ip),
    ]
    return dict(cls='suspicious', risk='MEDIUM', early=True, events=ev, conf=0.86,
                evidence=['device has no CMDB asset record', 'self-signed certificate on first connection'],
                analysis='A device that has never connected before appeared with no asset record and a self-signed certificate. Nothing destructive has happened, but an unmanaged endpoint reaching production is unexplained and could be a personal device or a foothold.',
                action='Ask the user to confirm the device, then quarantine it if unrecognised. Enrol the network in certificate-based admission so unknown devices cannot connect at all.')


@scenario('Suspicious behavior')
def dns_tunnel(rng, env, lens, when):
    host, n = rng.choice(ROLES), rng.randrange(30, 400)
    ev = [
        _ev(host, 'resolver', 6100, 'query TXT %s.dns-gateway-%d.example for record of type TXT, response 253 bytes' % (rng.choice(['a', 'b', 'c']), n), sev='warning', imp=7),
        _ev(host, 'resolver', 6101, 'query TXT %s.dns-gateway-%d.example repeated %d times in 90s from one process' % (rng.choice(['d', 'e', 'f']), n, n), sev='warning', imp=9),
        _ev(host, 'resolver', 6102, 'total TXT payload for this process %d KB, longest label 52 characters' % n, imp=10),
    ]
    return dict(cls='suspicious', risk='HIGH', early=True, events=ev, conf=0.83,
                evidence=['TXT queries to a nonsense subdomain at high rate', 'long labels and growing payload volume'],
                analysis='One process is issuing a rapid series of long TXT queries to an unrelated domain, which is how data is often smuggled out over DNS. The volume is still small, so this looks like staging rather than a completed transfer, but the channel is established.',
                action='Block the destination domain at the resolver, capture the query payloads for the record, and hunt the same process hash across hosts before assuming this is isolated.')


@scenario('Suspicious behavior')
def svc_account_widened(rng, env, lens, when):
    user = rng.choice(SVC_ACCOUNTS)
    ev = [
        _ev('adfs-01', 'audit', 3301, 'object %s modified: memberOf added group ROLE-DB-ADMIN-WRITEOPS' % user, sev='warning', imp=10, user=user),
        _ev('adfs-01', 'audit', 3302, 'privilege change for %s was made by %s using a role assignment, not an app approval workflow' % (user, rng.choice(HUMAN_USERS)), sev='warning', imp=9, user=user),
        _ev('db-primary', 'postgres', 3303, 'connection by %s now granted INSERT on schema finance' % user, sev='warning', imp=8, user=user),
    ]
    return dict(cls='suspicious', risk='HIGH', early=True, events=ev, conf=0.85,
                evidence=['service account added to a write-operations admin group', 'change bypassed the application approval workflow'],
                analysis='A service account that should be read-only was moved into a group granting write access to finance data, and the change was made directly in the directory rather than through the approval workflow. No misuse has been observed yet, but the account now has far more power than its function needs.',
                action='Revert the group membership now, then find out who made the change and whether it was requested. Add an alert on any group change that increases a service account privilege.')


@scenario('Suspicious behavior')
def mfa_disabled(rng, env, lens, when):
    user = rng.choice(HUMAN_USERS)
    ev = [
        _ev('adfs-01', 'audit', 4401, 'authentication method registered for %s changed: phone-based second factor removed, security key retained' % user, sev='warning', imp=9, user=user),
        _ev('adfs-01', 'audit', 4402, 'change performed by %s holding Helpdesk role, outside self-service' % rng.choice(HUMAN_USERS), sev='warning', imp=8),
        _ev('adfs-01', 'audit', 4403, '%s signed in from %s with password only, no second factor presented' % (user, rng.choice(IPS_EXT)), sev='warning', imp=10, user=user),
    ]
    return dict(cls='suspicious', risk='HIGH', early=True, events=ev, conf=0.86,
                evidence=['second factor removed by another admin', 'password-only sign-in immediately afterwards'],
                analysis='A helpdesk administrator removed a second factor from a user account and that same account then signed in with a password only from an external address. Either the user is being social-engineered through support or an insider removed the control deliberately.',
                action='Re-enrol the second factor, revoke active sessions for that account, and pull the helpdesk audit trail to see who approved the change and under what pretext.')


@scenario('Suspicious behavior')
def exfil_staging(rng, env, lens, when):
    host, size = rng.choice(ROLES), rng.randrange(2, 40)
    ev = [
        _ev(host, 'edr', 9901, 'archive created: %s.zip, %d MB, source directory %s' % (rng.choice(['export', 'dump', 'all_data']), size, rng.choice(PORTSHARE)), imp=8),
        _ev(host, 'proxy', 9902, 'outbound connection to cloud storage host attempted, %d MB upload over %d minutes' % (size, rng.randrange(2, 90)), sev='warning', imp=10),
        _ev(host, 'proxy', 9903, 'upload blocked by data-loss-prevention policy, no data left the network', sev='error', imp=9),
    ]
    return dict(cls='suspicious', risk='HIGH', early=True, events=ev, conf=0.84,
                evidence=['%d MB archive created from a file share' % size, 'outbound upload blocked by DLP'],
                analysis='A large archive was built from a file share and an upload to external cloud storage was attempted before the data-loss-prevention policy blocked it. Nothing left the network, so this is an interrupted attempt rather than a completed breach, but the intent is clear.',
                action='Isolate the host, capture the archive contents to establish what was collected, and review whether other hosts show the same compression-and-upload pattern.')


@scenario('Suspicious behavior')
def tor_exit(rng, env, lens, when):
    host, ip = rng.choice(ROLES), rng.choice(IPS_EXT)
    ev = [
        _ev(host, 'proxy', 2210, 'outbound TLS session to %s, certificate issuer matches a known Tor exit node list' % ip, sev='warning', imp=10, src=ip),
        _ev(host, 'proxy', 2211, 'same host opened %d further sessions to %d Tor exit addresses within 20 minutes' % (rng.randrange(2, 9), rng.randrange(2, 9)), sev='warning', imp=9),
        _ev(host, 'proxy', 2212, 'no hostname in SNI, session lengths average 8 minutes, traffic pattern matches a proxy client', sev='warning', imp=8),
    ]
    return dict(cls='suspicious', risk='HIGH', early=True, events=ev, conf=0.82,
                evidence=['connections to known Tor exit nodes', 'no SNI hostname, uniform session lengths'],
                analysis='A server is repeatedly tunnelling out through Tor exit nodes with no server name in the handshake, which is how command-and-control traffic is commonly disguised. On an infrastructure host with no stated reason to anonymise, this needs an explanation.',
                action='Block the exit-node list at the egress proxy, identify the owning process, and confirm whether that host is ever legitimately permitted outbound anonymity.')


@scenario('Suspicious behavior')
def geo_anomaly(rng, env, lens, when):
    user = rng.choice(HUMAN_USERS)
    c1, c2 = rng.sample(CITIES, 2)
    ev = [
        _ev('vpn-concentrator', 'vpn', 1101, '%s signed in from %s, MFA passed, device trusted' % (user, c1), imp=7, user=user),
        _ev('vpn-concentrator', 'vpn', 1102, '%s signed in from %s 41 minutes later, MFA passed, device NOT trusted' % (user, c2), sev='warning', imp=10, user=user),
        _ev('vpn-concentrator', 'vpn', 1103, 'second session came from %s, a new device fingerprint on a residential IP range' % c2, sev='warning', imp=9, user=user),
    ]
    return dict(cls='suspicious', risk='MEDIUM', early=True, events=ev, conf=0.87,
                evidence=['two sign-ins 41 minutes apart from distant cities', 'second device untrusted, residential IP'],
                analysis='One account authenticated twice in under an hour from two cities far enough apart to be impossible for the same person, and the second attempt came from a new device on a residential connection. The MFA step passed, which suggests the second factor was approved rather than skipped.',
                action='Verify with the user out of band before assuming compromise, revoke the unknown device, and require a trusted-device check for any session that follows a new fingerprint.')


# ----------------------------------------------- Early-warning sequences ----

@scenario('Early-warning sequences')
def slow_drift(rng, env, lens, when):
    host, user = rng.choice(ROLES), rng.choice(HUMAN_USERS)
    ev = [
        _ev(host, 'edr', 1201, '%s logged on interactively at an unusual hour for this account' % user, imp=6, user=user),
        _ev(host, 'edr', 1202, '%s added %d scheduled tasks, all named descriptively' % (user, rng.randrange(2, 6)), sev='warning', imp=7, user=user),
        _ev(host, 'edr', 1203, '%s enabled remote desktop for a local administrator account' % user, sev='warning', imp=8, user=user),
        _ev(host, 'edr', 1204, '%s read the security policy file, then exported the local group membership' % user, sev='warning', imp=9, user=user),
        _ev(host, 'edr', 1205, '%s attempted to create a new local administrator account, blocked by policy' % user, sev='error', imp=10, user=user),
    ]
    return dict(cls='suspicious', risk='HIGH', early=True, events=ev, conf=0.85,
                evidence=['unusual-hour interactive logon', 'RDP enabled on a local admin', 'account creation blocked by policy'],
                analysis='The sequence starts with an unremarkable off-hours logon and then walks steadily toward persistence and privilege: scheduled tasks, remote desktop on a local administrator account, reconnaissance of the security policy, and finally an account-creation attempt that policy blocked. The early signal is the off-hours logon, several steps before the decisive event.',
                action='Disable the account, preserve the full process timeline, and review whether the earlier tasks or exports were used. Alert on the off-hours logon as a precursor rather than waiting for the account-creation attempt.')


@scenario('Early-warning sequences')
def beacon_early(rng, env, lens, when):
    host = rng.choice(ROLES)
    ev = [
        _ev(host, 'proxy', 3301, 'outbound TLS to %s, 812 bytes, every 300s, jitter under 5 percent' % rng.choice(IPS_EXT), imp=7),
        _ev(host, 'proxy', 3302, 'same destination contacted %d times with identical size and interval' % rng.randrange(12, 200), imp=8),
        _ev(host, 'edr', 3303, 'process %s spawned by %s, neither matches installed software' % (rng.choice(PROCS), rng.choice(PROCS)), sev='warning', imp=9),
        _ev(host, 'edr', 3304, '%s attempted to disable the endpoint agent, access denied' % rng.choice(PROCS), sev='error', imp=10),
    ]
    return dict(cls='suspicious', risk='CRITICAL', early=True, events=ev, conf=0.88,
                evidence=['fixed-interval beacon with negligible jitter', 'unrecognised process ancestry', 'agent tampering attempt'],
                analysis='A fixed-size request every five minutes with almost no jitter is machine traffic, not a user. Following it back, an unrecognised process spawned another and then tried to disable the endpoint agent, which is textbook command-and-control with anti-forensic intent. The beacon appears two events before the tampering attempt.',
                action='Isolate the host immediately, sinkhole the destination, and hunt the whole estate for the same beacon interval. The agent-disable attempt is a strong enough signal to treat this as an active intrusion.')


@scenario('Early-warning sequences')
def ransomware_precursor(rng, env, lens, when):
    host = rng.choice(ROLES)
    ev = [
        _ev(host, 'edr', 4401, 'process %s executed with inline PowerShell encoded command' % rng.choice(['powershell.exe', 'pwsh.exe']), sev='warning', imp=8),
        _ev(host, 'edr', 4402, 'shadow copy deletion requested via %s, status denied' % rng.choice(['vssadmin.exe', 'wmic.exe']), sev='error', imp=9),
        _ev(host, 'edr', 4403, 'mass rename detected: %d files under %s changed extension' % (rng.randrange(200, 9000), rng.choice(PORTSHARE)), sev='critical', imp=10),
        _ev(host, 'edr', 4404, 'ransom note %s created in %d directories' % (rng.choice(['HOW_TO_DECRYPT.txt', 'README_FOR_DECRYPT.html']), rng.randrange(4, 60)), sev='critical', imp=10),
    ]
    return dict(cls='malicious', risk='CRITICAL', early=True, events=ev, conf=0.93,
                evidence=['shadow copy deletion denied', 'mass file rename across a file share', 'ransom notes written'],
                analysis='Encoded PowerShell ran, recovery was deliberately broken by a denied shadow-copy deletion, files were then mass-renamed and ransom notes written. This is ransomware executing end to end. The precursor is the encoded PowerShell, which is the earliest reliable indicator and appeared while the operator still had to be careless.',
                action='Treat as a confirmed ransomware incident: isolate the host and its reachable shares, disable the compromised account, preserve the note files, and engage the response plan now. Block the executable hashes estate-wide.')


# ---------------------------------------------------------- False positives ----

@scenario('False positives')
def change_ticket_burst(rng, env, lens, when):
    host, user = rng.choice(ROLES), rng.choice(HUMAN_USERS)
    ev = [
        _ev(host, 'firewall', 5501, 'rule widened: port %d opened from any to %s, change CHG-%d' % (rng.choice([22, 443, 3389]), host, rng.randrange(6000, 6999)), sev='warning', imp=8),
        _ev(host, 'firewall', 5502, 'concurrent connections to the newly opened port from %d sources' % rng.randrange(5, 90), sev='warning', imp=9),
        _ev(host, 'change', 5503, 'CHG-%d validated: temporary rule for migration, expiry %s set automatically' % (rng.randrange(6000, 6999), rng.choice(['02:00', '04:00', '06:00'])), imp=10),
    ]
    return dict(cls='normal', risk='LOW', early=False, events=ev, conf=0.92,
                evidence=['rule change carries an approved change record', 'automatic expiry set on the temporary rule'],
                analysis='A firewall rule that looks like an exposure was opened, but it carries a validated change record for an active migration and expires automatically. The traffic that followed is the migration doing its job, not an attacker using the opening.',
                action='Verify the rule disappears at its automatic expiry, and alert if it is still present afterwards. No incident response is warranted for the rule itself.')


@scenario('False positives')
def noisy_sensor(rng, env, lens, when):
    dev, n = rng.choice(ROLES), rng.randrange(200, 4000)
    ev = [
        _ev(dev, 'sensor-bridge', 6601, 'high-volume duplicate alert fired %d times for the same condition' % n, sev='warning', imp=7),
        _ev(dev, 'sensor-bridge', 6602, 'duplicate suppression engaged after %d repeats' % n, imp=9),
        _ev(dev, 'sensor-bridge', 6603, 'underlying condition still present, rate below alert threshold', sev='warning', imp=10),
    ]
    return dict(cls='normal', risk='LOW', early=False, events=ev, conf=0.9,
                evidence=['%d identical alerts from one sensor' % n, 'duplicate suppression engaged automatically'],
                analysis='The alert storm is a misbehaving sensor re-reporting one condition thousands of times, and the platform already suppressed the duplicates. The single remaining alert is a real but low-severity condition on one device, not a fleet-wide problem.',
                action='Replace or recalibrate the sensor and tune the rule to suppress by condition rather than by count. Keep the single underlying alert open at its genuine severity.')


@scenario('False positives')
def backup_growth(rng, env, lens, when):
    host, x = rng.choice(ROLES), rng.randrange(3, 30)
    ev = [
        _ev(host, 'backup-agent', 7701, 'backup volume %d GB, %dx the 30-day mean' % (rng.randrange(200, 3000), x), sev='warning', imp=8),
        _ev(host, 'backup-agent', 7702, 'delta reason: first full backup after retention change on %s' % rng.choice(['fileshare', 'imaging', 'cad-archive']), imp=9),
        _ev(host, 'backup-agent', 7703, 'post-change baseline recalculated, growth within expected envelope', imp=10),
    ]
    return dict(cls='normal', risk='LOW', early=False, events=ev, conf=0.91,
                evidence=['growth explained by a retention change', 'baseline recalculated and within envelope'],
                analysis='Backup volume multiplied by a large factor, but the agent attributes it to the first full backup after a retention change, which legitimately copies far more data than an incremental run. The recalculated baseline confirms the size is expected.',
                action='No response. Keep watching the next incremental run to confirm volume returns to normal, and adjust the growth alert threshold so planned retention changes stop firing it.')


@scenario('False positives')
def pentest_kickoff(rng, env, lens, when):
    user, ip = rng.choice(HUMAN_USERS), rng.choice(IPS_EXT)
    ev = [
        _ev('auth-gateway', 'sshd', 7801, 'Failed password for root from %s' % ip, sev='warning', imp=7, src=ip),
        _ev('auth-gateway', 'sshd', 7802, 'Failed password for admin from %s' % ip, sev='warning', imp=7, src=ip),
        _ev('auth-gateway', 'proxy', 7803, 'engagement %s started under change record, source whitelisted for the test window' % rng.choice(['pentest', 'red-team', 'assurance']), imp=10),
    ]
    return dict(cls='suspicious', risk='MEDIUM', early=True, events=ev, conf=0.89,
                evidence=['brute-force signature matches an authorised engagement', 'source whitelisted for the test window'],
                analysis='Failed root and admin logins from one address look exactly like a credential attack, and the alerting is right to fire. The source is whitelisted for an authorised security assessment that starts in this window, which is the documented benign explanation.',
                action='No breach response. Keep the alerts visible so the engagement has a record, and confirm the whitelist entry is removed the moment the test window closes.')


# ----------------------------------------------- Multi-signal correlation ----

@scenario('Multi-signal correlation')
def benign_alone_malicious_together(rng, env, lens, when):
    a, b, user = rng.choice(ROLES), rng.choice(ROLES), rng.choice(SVC_ACCOUNTS)
    ev = [
        _ev(a, 'proxy', 8801, 'outbound connection to %s, first ever destination for this host' % rng.choice(IPS_EXT), imp=5),
        _ev(a, 'edr', 8802, 'new file %s.exe written to a startup folder, unsigned' % rng.choice(PROCS), imp=6),
        _ev(b, 'auth', 8803, '%s authenticated from %s which is not in the approved service host list' % (user, rng.choice(IPS_INT)), imp=6),
        _ev(a, 'proxy', 8804, 'same destination contacted again with %d KB, now from a second host %s' % (rng.randrange(50, 900), b), imp=8),
        _ev(a, 'edr', 8805, 'the unsigned startup file and the outbound destination resolve to the same certificate', sev='error', imp=10),
    ]
    return dict(cls='malicious', risk='CRITICAL', early=True, events=ev, conf=0.9,
                evidence=['no single event is decisive alone', 'new unsigned startup persistence', 'outbound traffic from two hosts to one new destination', 'file and destination share a certificate'],
                analysis='Individually each event is weak: one first-time destination, one unsigned startup file, one service account from an unapproved host. Together they describe a shared toolchain, and the certificate match between the dropped file and the remote endpoint is what turns three signals into one conclusion. No single alert would have justified escalation.',
                action='Isolate both hosts, treat them as one incident rather than two, and search the estate for the shared certificate. Correlation like this should raise the alert, not wait for a single-tool match.')


@scenario('Multi-signal correlation')
def asset_drift(rng, env, lens, when):
    host, ip = rng.choice(ROLES), rng.choice(IPS_INT)
    ev = [
        _ev(host, 'edr', 8901, 'operating system build differs from the golden image for this role', imp=5),
        _ev(host, 'edr', 8902, 'antivirus real-time protection reported off for %d hours' % rng.randrange(4, 90), imp=6),
        _ev(host, 'config', 8903, 'host contacted a configuration server outside its declared subnet boundary', imp=6),
        _ev('cmdb', 'asset-sync', 8904, '%s is assigned to subnet %s but last seen on %s' % (host, rng.choice(IPS_INT), ip), sev='warning', imp=8),
        _ev('cmdb', 'asset-sync', 8905, 'decommission record exists for this asset, last verified %d days ago' % rng.randrange(30, 400), sev='warning', imp=9),
    ]
    return dict(cls='suspicious', risk='HIGH', early=True, events=ev, conf=0.86,
                evidence=['build drift from golden image', 'endpoint protection disabled for hours', 'asset marked decommissioned but still active'],
                analysis='Each fact alone is a ticket: an unpatched build, antivirus off, a subnet move. The pattern that matters is that the asset is recorded as decommissioned yet is still live and talking to a new subnet. A retired machine that is still reachable is often how access outlives an offboarding, and the other signals make it more likely to be in use.',
                action='Confirm with the decommission record who still needs this host, isolate it pending that answer, and audit the change process that let a retired asset stay live.')


# --------------------------------------------------------- Authentication ----

@scenario('Authentication')
def legacy_protocol_auth(rng, env, lens, when):
    host, user = rng.choice(ROLES), rng.choice(HUMAN_USERS)
    ev = [
        _ev(host, 'winsec', 9001, '4625 logon failure for %s, account %s, logon type 3, source %s' % (user, rng.choice(HUMAN_USERS), rng.choice(IPS_INT)), imp=6, winid=4625),
        _ev(host, 'winsec', 9002, '4771 pre-authentication failed for %s, requesting server %s' % (user, rng.choice(ROLES)), imp=6, winid=4771),
        _ev(host, 'winsec', 9003, '4625 logon failure, NTLM only, account not configured for Kerberos', imp=7, winid=4625),
        _ev(host, 'ntlm', 9004, 'inbound NTLM authentication accepted from %s for %d accounts' % (rng.choice(IPS_EXT), rng.randrange(2, 40)), sev='warning', imp=9),
    ]
    return dict(cls='suspicious', risk='MEDIUM', early=True, events=ev, conf=0.87,
                evidence=['NTLM-only authentication accepted inbound', 'Kerberos pre-authentication failing', 'downgrade to NTLM for multiple accounts'],
                analysis='Kerberos pre-authentication is failing and the machine is falling back to NTLM, which is accepted for several accounts including from an external range. This reads as a legacy client or a deliberate downgrade, and it exposes credentials that can be relayed.',
                action='Require Kerberos for the domain and disable inbound NTLM where the client inventory allows. Until then, harden channel binding so relayed NTLM cannot be used.')


@scenario('Authentication')
def service_account_anomaly(rng, env, lens, when):
    user = rng.choice(SVC_ACCOUNTS)
    ev = [
        _ev('adfs-01', 'audit', 9101, 'sign-in for %s from %s using interactive token' % (user, rng.choice(IPS_EXT)), sev='warning', imp=9, user=user),
        _ev('adfs-01', 'audit', 9102, '%s is configured for a non-interactive logon type only; interactive use is anomalous' % user, imp=8, user=user),
        _ev('adfs-01', 'audit', 9103, 'same %s session issued %d directory queries across %d different OUs' % (user, rng.randrange(40, 900), rng.randrange(5, 60)), sev='warning', imp=10, user=user),
    ]
    return dict(cls='suspicious', risk='HIGH', early=True, events=ev, conf=0.88,
                evidence=['service account used interactively', 'broad directory enumeration afterwards'],
                analysis='A service account that is only ever supposed to run non-interactively signed in with an interactive token and then enumerated large parts of the directory. Service accounts are a favourite target precisely because nobody watches their sign-ins, so this is worth escalating even though no damage is visible.',
                action='Reset the account credentials, review every action it performed this session, and alert on any service account that ever authenticates interactively.')


@scenario('Authentication')
def successful_then_drift(rng, env, lens, when):
    user, ip = rng.choice(HUMAN_USERS), rng.choice(IPS_EXT)
    ev = [
        _ev('vpn-concentrator', 'vpn', 9201, '%s authenticated with password and MFA, assigned pool 10.31.0.0/16' % user, imp=6, user=user),
        _ev('vpn-concentrator', 'proxy', 9202, 'assigned address %s made a request to a file share outside the group the user is entitled to' % rng.choice(IPS_INT), sev='warning', imp=8, user=user),
        _ev('vpn-concentrator', 'edr', 9203, 'on the assigned address, an attempt to read %s which is not in the user entitlement report' % rng.choice(PORTSHARE), sev='warning', imp=9, user=user),
        _ev('vpn-concentrator', 'edr', 9204, 'attempt logged and allowed by the share ACL, so no technical control stopped it', imp=10, user=user),
    ]
    return dict(cls='suspicious', risk='MEDIUM', early=True, events=ev, conf=0.84,
                evidence=['access outside the user entitlement report', 'share ACL allows what policy should not'],
                analysis='Authentication itself was clean, so this would not appear in a login anomaly report. What matters is what the session did afterwards: reaching a share the user is not entitled to, and the share ACL permitting it. The identity is probably compromised rather than the account misused, and the ACL gap is a real finding on its own.',
                action='Force a password reset and revoke sessions, then correct the share ACL against the entitlement report. Review the same path for other users, since the ACL problem will not be unique.')


# ---------------------------------------------------------------- Network ----

@scenario('Network')
def dns_beaconing(rng, env, lens, when):
    host, dest = rng.choice(ROLES), rng.choice(IPS_EXT)
    ev = [
        _ev(host, 'dns', 9301, 'repeated A-record lookups for %s.dns-cdn-%d.net, NXDOMAIN %d times' % (rng.choice(['a', 'b', 'c']), rng.randrange(10, 99), rng.randrange(20, 300)), imp=6),
        _ev(host, 'dns', 9302, 'subdomain label lengths average 38 characters, above the 24-character baseline', imp=7),
        _ev(host, 'dns', 9303, 'the host resolved %d unique subdomains of this domain, all NXDOMAIN' % rng.randrange(30, 900), sev='warning', imp=9),
        _ev(host, 'dns', 9304, 'no other internal host has ever queried this domain', sev='warning', imp=10),
    ]
    return dict(cls='suspicious', risk='HIGH', early=True, events=ev, conf=0.85,
                evidence=['long NXDOMAIN subdomain labels', 'hundreds of unique subdomains from one host', 'no other host queries the domain'],
                analysis='One host is generating an unbounded stream of unique, long, non-existent subdomains under a single external domain. Nothing resolves, which means the queries themselves are the channel. That is the standard shape of DNS tunnelling used for command and control or slow exfiltration.',
                action='Sinkhole the domain and isolate the host, then decode the query payloads to establish what left the network. Block the domain resolver-wide so no other host can be added later.')


@scenario('Network')
def lateral_smb(rng, env, lens, when):
    src, n = rng.choice(IPS_INT), rng.randrange(5, 120)
    ev = [
        _ev(src, 'smb', 9401, 'session setup to %d hosts on the same subnet, administrative share requested' % n, imp=7),
        _ev(src, 'smb', 9402, 'share enumeration on %d hosts, ADMIN$ accessed on %d of them' % (n, rng.randrange(1, 12)), sev='warning', imp=8),
        _ev(src, 'smb', 9403, 'service creation attempted on %d remote hosts, psexec-style named pipe' % rng.randrange(1, 12), sev='error', imp=10),
    ]
    return dict(cls='malicious', risk='CRITICAL', early=True, events=ev, conf=0.9,
                evidence=['administrative share sweep across the subnet', 'remote service creation named-pipe pattern'],
                analysis='A host enumerated administrative shares across its subnet and then attempted to create a service through a named pipe on several of them, which is the standard remote-execution pattern for lateral movement. Administrative share access to many hosts at once from a single workstation has no benign equivalent.',
                action='Isolate the source host and treat every touched machine as compromised until cleared. Reset the service account it used, and block administrative shares at the network level between workstations.')


@scenario('Network')
def port_scan_legit_host(rng, env, lens, when):
    host = rng.choice(ROLES)
    ev = [
        _ev(host, 'ids', 9501, 'TCP SYN to %d ports on this host in 4 seconds from %s' % (rng.randrange(10, 60), rng.choice(IPS_INT)), sev='warning', imp=6),
        _ev(host, 'ids', 9502, 'the scanning source is the network monitoring appliance, which inventories weekly', imp=9),
        _ev(host, 'ids', 9503, 'all probed ports are documented services, no unexpected listener found', imp=10),
    ]
    return dict(cls='normal', risk='INFO', early=False, events=ev, conf=0.93,
                evidence=['scan source is the registered inventory appliance', 'probed ports match documented services'],
                analysis='A burst of connection attempts against many ports is the inventory job running from the monitoring appliance, which is scheduled to sweep weekly. The scan found only documented listeners, which confirms the host matches its intended configuration.',
                action='No action. Tune the IDS signature to suppress the inventory source, and confirm the sweep is still running on its normal weekly cadence.')


# ------------------------------------------------------- Process behavior ----

@scenario('Process behavior')
def process_injection(rng, env, lens, when):
    host, good, bad = rng.choice(ROLES), rng.choice(PROCS), rng.choice(['svchost.exe', 'explorer.exe', 'notepad.exe'])
    ev = [
        _ev(host, 'edr', 9601, '%s started with parent %s and command line referencing a remote share' % (bad, rng.choice(PROCS)), sev='warning', imp=7),
        _ev(host, 'edr', 9602, 'process %s opened %s and wrote an executable to a temp path' % (bad, rng.choice(PORTSHARE)), sev='warning', imp=8),
        _ev(host, 'edr', 9603, '%s called %s with WRITE_PROC_MEM, a process injection primitive' % (bad, good), sev='error', imp=10),
        _ev(host, 'edr', 9604, 'target process %s now hosts a thread with a different image base' % good, sev='critical', imp=10),
    ]
    return dict(cls='malicious', risk='CRITICAL', early=True, events=ev, conf=0.92,
                evidence=['temp-path executable write from a remote share', 'WRITE_PROC_MEM into a trusted binary'],
                analysis='A process launched from a remote share wrote an executable to a temp path and then opened another process with write access to its memory, which is process injection. The thread running inside the trusted binary is the payload, and the trusted name is what makes it survive casual inspection.',
                action='Kill the host, capture memory before reboot if forensics value the session, and hunt the temp-path binary and the injected process pair across all endpoints. Block the remote share path that started it.')


@scenario('Process behavior')
def odd_but_legitimate_tool(rng, env, lens, when):
    host, user = rng.choice(ROLES), rng.choice(HUMAN_USERS)
    ev = [
        _ev(host, 'edr', 9701, '%s launched %s with an encoded command line, flagged by heuristic rule' % (rng.choice(PROCS), rng.choice(PROCS)), sev='warning', imp=6, user=user),
        _ev(host, 'edr', 9702, 'decode shows a documented build script reading a version file and invoking the artifact repository API', imp=9, user=user),
        _ev(host, 'edr', 9703, 'the script is signed, version-matched and present in the change record for this build', imp=10, user=user),
    ]
    return dict(cls='normal', risk='LOW', early=False, events=ev, conf=0.9,
                evidence=['encoded command matches a signed build script', 'script version-matched to an active change record'],
                analysis='An encoded PowerShell command triggered a heuristic alert, but decoding it shows the standard build script that reads a version file and calls the artifact repository. It is signed, version-matched and named in the current change record, so the encoding is tooling hygiene rather than evasion.',
                action='No response. Consider allow-listing the signed script by hash to stop the heuristic firing on every build, which would otherwise train the team to ignore it.')


@scenario('Process behavior')
def scheduled_task_persistence(rng, env, lens, when):
    host, user = rng.choice(ROLES), rng.choice(HUMAN_USERS)
    ev = [
        _ev(host, 'edr', 9801, 'scheduled task %s created by %s, runs at logon with highest privileges' % (rng.choice(['OneDriveSync', 'SystemUpdate', 'HealthCheck']), user), sev='warning', imp=8, user=user),
        _ev(host, 'edr', 9802, 'task action is %s launched from a per-user appdata path' % rng.choice(PROCS), sev='warning', imp=9),
        _ev(host, 'edr', 9803, 'the referenced binary is unsigned and was created %d minutes earlier' % rng.randrange(1, 60), sev='error', imp=10),
        _ev(host, 'edr', 9804, 'task fired on the next logon and spawned an outbound connection', sev='critical', imp=10),
    ]
    return dict(cls='malicious', risk='CRITICAL', early=True, events=ev, conf=0.9,
                evidence=['privileged logon task with a benign-sounding name', 'unsigned binary from appdata created minutes before', 'task produced outbound traffic'],
                analysis='A scheduled task named to look like a healthy Windows component runs at logon with the highest privilege level, and its payload is an unsigned binary dropped into a per-user appdata path minutes earlier. It has already executed and dialled out. The naming is the disguise; the unsigned recent binary is the giveaway.',
                action='Isolate the host, remove the task and the binary, and reset the account. Alert on privileged scheduled tasks referencing appdata, and hunt for other tasks using trusted-sounding names.')


# -------------------------------------------- Vulnerability / code security ----

@scenario('Vulnerability / code security')
def injection_orm(rng, env, lens, when):
    frag = rng.choice([
        'q = "SELECT * FROM orders WHERE customer = \'%s\'" % cust',
        'db.execute("SELECT * FROM users WHERE name = \'" + name + "\'")',
        'sql = f"DELETE FROM sessions WHERE token = \'{token}\'"',
    ])
    ev = [
        _ev('waf-edge-01', 'waf', 9901, 'request to /api/%s contained %d consecutive single quotes in a parameter' % (rng.choice(['orders', 'users', 'cart']), rng.randrange(4, 20)), sev='warning', imp=7),
        _ev('waf-edge-01', 'app', 9902, 'handler built a query by string interpolation: %s' % frag, sev='error', imp=10),
        _ev('app-error', 'error', 9903, 'database rejected the statement, connection returned to the pool without being discarded', sev='error', imp=9),
    ]
    return dict(cls='malicious', risk='CRITICAL', early=True, events=ev, conf=0.91,
                evidence=['input reaches a SQL string built by interpolation', 'connection returned to the pool after a database error'],
                analysis='A request parameter carrying stacked quotes reaches a query assembled by string interpolation, which is a textbook SQL injection sink. Returning the connection to the pool after the database error is the second defect: a pooled connection can carry a transaction state into the next unrelated request. The flaw is in the handler, quoted here so the fix is unambiguous.',
                action='Replace the interpolation with a parameterised query, return the connection to the pool only after a rollback, and add a test that submits a quote-bearing parameter. Review every sibling handler for the same pattern.')


@scenario('Vulnerability / code security')
def deserialization(rng, env, lens, when):
    ev = [
        _ev('app', 'waf', 9911, 'POST to /api/callback carried a base64 blob of %d KB in the state parameter' % rng.randrange(4, 90), sev='warning', imp=7),
        _ev('app', 'app', 9912, 'endpoint deserialises the state parameter with a polymorphic object deserialiser', sev='error', imp=10),
        _ev('app', 'app', 9913, 'the process spawned a child interpreter and attempted an outbound connection before the request returned', sev='critical', imp=10),
    ]
    return dict(cls='malicious', risk='CRITICAL', early=True, events=ev, conf=0.9,
                evidence=['polymorphic deserialisation of a client-supplied blob', 'child interpreter spawned during the request'],
                analysis='A large base64 blob arrives in a state parameter and is handed to a polymorphic object deserialiser, which is remote code execution the moment an attacker supplies a gadget chain. The child interpreter spawning during the request is the exploit succeeding, not a theoretical concern. This is the defect, described rather than demonstrated.',
                action='Stop deserialising client input entirely. If a state token is required, use a signed, short-lived, self-contained format, rotate any secret that could serve as a signing key, and assume the process is compromised for this request path.')


@scenario('Vulnerability / code security')
def dependency_rce(rng, env, lens, when):
    cve, pkg = rng.choice(CVE), rng.choice(['xz-utils', 'libssh', 'openssl', 'glibc', 'curl'])
    ev = [
        _ev('build-runner', 'sbom', 9921, 'dependency %s resolved to a version affected by %s' % (pkg, cve), sev='warning', imp=7),
        _ev('build-runner', 'sca', 9922, 'the affected code path is reachable when parsing attacker-supplied compressed input', sev='error', imp=9),
        _ev('build-runner', 'sca', 9923, 'built artefact ships this library into an internet-facing service', sev='critical', imp=10),
    ]
    return dict(cls='suspicious', risk='HIGH', early=True, events=ev, conf=0.89,
                evidence=['known-vulnerable library version in the SBOM', 'vulnerable path reachable with untrusted input', 'artefact deployed internet-facing'],
                analysis='The build resolved a library version that is affected by a published remote code execution flaw, the vulnerable code path is reachable from untrusted input, and the resulting artefact is running on an internet-facing service. Nothing has been exploited here, so this is a warning, but the exposure window is open right now.',
                action='Rebuild on a patched version, redeploy, and pin the dependency with a version floor in the build manifest. Add an SBOM gate to the pipeline so this is caught before release rather than by a scan afterwards.')


@scenario('Vulnerability / code security')
def weak_crypto_config(rng, env, lens, when):
    ev = [
        _ev('waf-edge-01', 'config', 9931, 'TLS configuration permits %s, considered broken' % rng.choice(['TLS 1.0', 'TLS 1.1', '3DES', 'RC4']), sev='warning', imp=7),
        _ev('waf-edge-01', 'config', 9932, 'cipher suite list allows anonymous or export-grade ciphers', sev='warning', imp=8),
        _ev('waf-edge-01', 'config', 9933, 'the same profile is inherited by %d downstream services' % rng.randrange(3, 40), sev='error', imp=9),
    ]
    return dict(cls='suspicious', risk='MEDIUM', early=True, events=ev, conf=0.88,
                evidence=['broken protocol version permitted', 'export-grade ciphers allowed', 'profile inherited across many services'],
                analysis='The edge accepts protocol versions and ciphers that are known to be broken, and because the setting is inherited the same weakness is copied into dozens of downstream services. There is no sign of exploitation, but every one of those services is one downgrade away from disclosure.',
                action='Set a modern minimum TLS version, drop the export ciphers, and roll the corrected profile to the services that inherit it. Monitor for handshake failures afterwards to catch anything relying on the old settings.')


# ---------------------------------------------------------------- Unknown ----

@scenario('Unknown')
def insufficient_evidence(rng, env, lens, when):
    host = rng.choice(ROLES)
    ev = [
        _ev(host, 'edr', 9941, 'anomaly score %d raised, contributing signals not retained by the sensor' % rng.randrange(60, 95), imp=6),
        _ev(host, 'edr', 9942, 'process ancestry unavailable, the parent exited before the snapshot was taken', imp=6),
        _ev(host, 'edr', 9943, 'no file, network or authentication correlation available for the time range', imp=7),
    ]
    return dict(cls='unknown', risk='MEDIUM', early=False, events=ev, conf=0.55,
                evidence=['anomaly raised but the underlying signals were not retained'],
                analysis='The sensor raised an anomaly score but discarded the evidence behind it, and the process ancestry is unavailable, so the finding can be neither confirmed nor dismissed. Calling this suspicious would be a guess, and calling it normal would be a worse one, so the honest classification is unknown until the data is recovered.',
                action='Do not close this. Retrieve the sensor detail for the time range, fix the retention setting that discarded the contributing signals, and re-evaluate once the process ancestry can be reconstructed.')


# ----------------------------------------------------------------------------------
# Generation.
# ----------------------------------------------------------------------------------

MIN_INPUT_CHARS, MAX_INPUT_CHARS = 120, 480
MIN_ANALYSIS_CHARS, MAX_ANALYSIS_CHARS = 40, 420
MIN_ACTION_CHARS = 40
MAX_CONFIDENCE = 0.97
IMPORTANCE_FLOOR = 3


def fit_input(events, renderer, rng, budget=MAX_INPUT_CHARS, floor=IMPORTANCE_FLOOR):
    """Render events, keeping the highest-importance ones that fit inside the budget.

    Events are always emitted in chronological order, and at least IMPORTANCE_FLOOR of
    them survive so a record never degenerates to a single line.
    """
    keep = [e for e in events if e.get('imp', 5) >= floor] or list(events)
    keep.sort(key=lambda e: -e.get('imp', 5))
    chosen = []
    for e in keep:
        trial = chosen + [e]
        trial.sort(key=lambda e: (e.get('ts', 0), e.get('order', 0)))
        text = ' | '.join(renderer(dict(x), rng) for x in trial)
        if len(text) <= budget:
            chosen = trial
        if len(chosen) >= 8:
            break
    if not chosen:
        chosen = list(events)[:1]
    chosen.sort(key=lambda e: (e.get('ts', 0), e.get('order', 0)))
    return ' | '.join(renderer(dict(x), rng) for x in chosen)


def validate(rec, input_text):
    """Return None if the record is usable, else a short reason. Mirrors the teacher gate."""
    if not (MIN_INPUT_CHARS <= len(input_text) <= MAX_INPUT_CHARS):
        return 'input_len'
    if not (MIN_ANALYSIS_CHARS <= len(rec['analysis']) <= MAX_ANALYSIS_CHARS):
        return 'analysis_len'
    if not (MIN_ACTION_CHARS <= len(rec['recommended_action']) <= MAX_ANALYSIS_CHARS):
        return 'action_len'
    if rec['classification'] not in ('normal', 'suspicious', 'malicious', 'unknown'):
        return 'bad_class'
    if rec['risk'] not in ('INFO', 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'):
        return 'bad_risk'
    if rec['early_warning'] != (rec['classification'] in ('suspicious', 'malicious')):
        return 'early_inconsistent'
    if not rec['evidence']:
        return 'no_evidence'
    if is_repetitive(input_text) or is_repetitive(rec['analysis']):
        return 'repetitive'
    return None


REPEAT_ANCHOR, MAX_REPEATS = 60, 3


def is_repetitive(text, anchor=REPEAT_ANCHOR, max_repeats=MAX_REPEATS, window=MAX_INPUT_CHARS):
    """True when a long span repeats, i.e. the text looped or templated into uselessness."""
    if len(text) < anchor * 2:
        return False
    for off in (anchor // 3, anchor // 2, 2 * anchor // 3):
        a = text[off:off + anchor]
        if len(a) == anchor and text.count(a) > max_repeats:
            return True
    return False


def make_record(rng, category, env, lens, when, variant):
    """Compose one record. Returns (record_dict, input_text) or (None, reason)."""
    pool = SCENARIOS[category]
    builder = rng.choice(pool)
    raw = builder(rng, env, lens, when)

    # The same analysis reads differently depending on where and when it is reviewed, so
    # the grid conditions the wording. Without this the corpus carries one fixed analysis
    # per scenario and the model memorises a few dozen sentences instead of learning to
    # reason. Pass None for any dimension to skip that conditioning.
    analysis, action = contextualise(raw['analysis'], raw['action'], env, when, variant, rng)

    # Stamp a plausible timestamp and stable ordering so renders are chronological.
    m, d, h = rng.randrange(1, 13), rng.randrange(1, 29), rng.randrange(24)
    base = rng.randrange(3600)
    ordered = []
    for i, e in enumerate(raw['events']):
        e['m'], e['d'] = m, d
        e['h'] = (h + (base // 3600)) % 24
        e['mi'] = (base // 60) % 60
        e['s'] = (base + i * rng.randrange(1, 90)) % 60
        e.setdefault('ts', i)
        e.setdefault('order', i)
        ordered.append(e)

    renderer = rng.choice(RENDERERS)
    text = fit_input(ordered, renderer, rng)

    record = {
        'input': text,
        'classification': raw['cls'],
        'risk': raw['risk'],
        'early_warning': raw['early'],
        'evidence': list(raw['evidence']),
        'analysis': analysis,
        'recommended_action': action,
        'confidence': min(MAX_CONFIDENCE, float(raw['conf'])),
        'hardness': {'INFO': 5, 'LOW': 20, 'MEDIUM': 45, 'HIGH': 70, 'CRITICAL': 90}[raw['risk']],
        'category': category,
    }
    reason = validate(record, text)
    if reason:
        return None, reason
    return record, text


# ======================================================================================
# Contextual assembly.
#
# Each scenario carries one fixed analysis and one fixed action, which on their own would
# mean 38 distinct response strings across the whole corpus - the model would memorise them
# instead of learning to reason. The grid dimensions (environment, timing, review context)
# are used to append a neutral contextual sentence, so the reasoning stays intact while the
# phrasing varies. The appended clauses are deliberately factual about the setting and never
# about the scenario, so they cannot contradict the label.
# ======================================================================================



ENV_CLAUSES = [
    'The cluster shares hardware with other tenants, so assume the blast radius is wider than the affected namespace.',
    'Clinical devices here cannot take a normal patch cycle, which limits the containment options.',
    'Registration week brings tens of thousands of short-lived accounts, so raw account volume proves nothing on its own.',
    'Cardholder data sits behind this path, so the notification clock starts at first indication, not at confirmation.',
    'The control segment cannot absorb an aggressive network response without risking the process itself.',
    'The VPN is how responders reach this estate, so a block there hits the response as well as the subject.',
    'One mistake here crosses a customer boundary, so tenant attribution matters more than the alert count.',
    'Every request already passes an identity-aware proxy, so a decision made without that context is working blind.',
    'Telematics units report over intermittent links, so gaps in the sequence are expected and are not evidence by themselves.',
    'Remote stations poll slowly, so the available data is minutes behind what is actually happening.',
    'The document store is externally hosted, so a contractual notification window may already be running.',
    'Render workers touch large shared volumes constantly, so volume alone is not a signal here.',
]

TIMING_CLAUSES = [
    'Staff are on shift, so a second opinion is available within minutes.',
    'Overnight cover is thin, so any decision now is likely to be made without a reviewer.',
    'Only skeleton cover is on duty, which makes escalation both expensive and easy to postpone.',
    'A deployment just finished, so a change-related explanation is the default assumption for the next few minutes.',
    'Backup traffic is expected to be heavy, so volume-based signals are unreliable right now.',
    'A rule change just landed, so recent connection counts are not a usable baseline.',
    'Billing activity is peaking, which explains part of the traffic and none of the rest.',
    'The vendor is mid-maintenance, which accounts for some of the change and leaves the rest unexplained.',
]

VARIANT_CLAUSES = [
    'With no second opinion available, the write-up has to carry the reasoning by itself.',
    'Two reviewers disagree on significance, so what gets recorded matters as much as the verdict.',
    'The outgoing analyst got this wrong earlier, so an explicit re-check is worth more than a fresh signature.',
    'The reviewer has no access to the originating request, so intent can only be inferred from behaviour.',
    'An auditor will reconstruct this from logs alone later, so the ordering and timestamps have to hold up.',
    'The timeline matters more than any single alert, so getting the sequence right is the priority.',
    'The live question is what has to be reported and by when, not only what happened.',
    'This is a week-later follow-up, so persistence and repeat behaviour matter more than the first signal.',
    'The same identity has done this before, which changes both the baseline and the likely response.',
    'Nothing about this identity has been seen before, so there is no history to calibrate against.',
    'The evidence is incomplete and that is itself the finding, so the honest answer is not a confident one.',
    'Three tools flagged different things, so the correlation between them is the whole case.',
    'A change ticket proves the benign reading, which is worth stating outright rather than inferring.',
    'The benign explanation is plausible and wrong, which is the case that most often gets closed by default.',
    'Severity here is set by the asset rather than the behaviour, so identifying the asset is the actual decision.',
    'The same signature means different things on two hosts, so host identity is load-bearing.',
    'The alert fired hours before anything happened, so the useful output is the lead time.',
    'Nothing has happened yet, so the value is in the next step rather than in the conclusion.',
    'The user is a contractor with narrower expected access, which makes an ordinary action notable.',
    'Service and human accounts are easily conflated here, and the distinction changes the response entirely.',
    'A scheduled job and an intruder look identical in the logs, so provenance has to be proven, not assumed.',
    'The signal is an absence rather than an addition, which is harder to notice and easier to dismiss.',
    'The activity is staying inside the normal baseline deliberately, so the absence of a threshold breach means little.',
    'Escalating is correct here rather than blocking, so the write-up should say why containment is premature.',
]

LEADS = ['Context matters: %s', 'Read against that setting: %s', 'Weighed in context - %s',
         'What changes the read: %s', 'Bearing that in mind: %s']

ACTION_TAILS = [
    'Record the decision and the reasoning in the case file so the next reviewer inherits the context.',
    'Log the outcome against the change record, whether or not it turned out to be related.',
    'Give this a named owner and a review date rather than closing it as reviewed.',
    'Note the false-positive outcome in the rule tuning backlog so the same alert is cheaper next time.',
    'Attach the raw events to the case so the conclusion can be re-checked later.',
    'Re-test the control that would have caught this and confirm it still fires.',
    'Write down what would have changed the outcome, so the detection gap is explicit.',
    'Confirm with the asset owner that the behaviour seen is the behaviour expected.',
    'Keep the timeline intact and hand it to whoever reviews this next.',
    'If it recurs, correlate on the identity rather than the signature next time.',
    'Raise the finding with the platform team, since the control gap outlives this one alert.',
    'Preserve the evidence before any remediation, because remediation overwrites the useful part.',
    'Set a watch on this identity for a week rather than a one-time action.',
    'Check whether the same pattern appears on the peers before declaring it isolated.',
    'Document the benign explanation explicitly, so the next occurrence is not re-investigated from scratch.',
    'Verify the fix in the environment that matters, not just in the test that passed.',
]


def _fit(base, options, limit):
    """Append the first option that keeps the text within *limit*."""
    base = base.rstrip()
    if base and not base.endswith(('.', '!', '?')):
        base += '.'
    for opt in options:
        if not opt:
            continue
        cand = base + ' ' + opt
        if len(cand) <= limit:
            return cand
    return base


def _clause_order(rng, parts):
    parts = [p for p in parts if p]
    rng.shuffle(parts)
    return ' '.join(parts)


def contextualise(analysis, action, env, when, variant, rng):
    """Return (analysis, action) with a grid-conditioned sentence appended to each."""
    envc = ENV_CLAUSES[ENVIRONMENTS.index(env)] if env in ENVIRONMENTS else rng.choice(ENV_CLAUSES)
    tmc = TIMING_CLAUSES[TIMINGS.index(when)] if when in TIMINGS else rng.choice(TIMING_CLAUSES)
    vrc = (VARIANT_CLAUSES[VARIANTS.index(variant)]
           if variant in VARIANTS else rng.choice(VARIANT_CLAUSES))

    triples = _clause_order(rng, [envc, tmc, vrc])
    pairs = [_clause_order(rng, [envc, tmc]), _clause_order(rng, [envc, vrc]),
             _clause_order(rng, [tmc, vrc])]
    singles = [envc, tmc, vrc]
    opts = []
    for body in [triples] + pairs + singles:
        lead = rng.choice(LEADS)
        opts.append(lead % body)
    analysis = _fit(analysis, opts, MAX_ANALYSIS_CHARS)

    tails = rng.sample(ACTION_TAILS, 3)
    action = _fit(action, [tails[0], _clause_order(rng, [tails[0], tails[1]]), tails[1]], MAX_ANALYSIS_CHARS)
    return analysis, action
