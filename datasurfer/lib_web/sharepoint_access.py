import requests
import os
import ssl
import tempfile
from requests_kerberos import HTTPKerberosAuth, OPTIONAL

class SharepointAccess(object):
    
    def __init__(self, url):
        """
        Initializes a SharepointAccess object.

        Args:
            url (str): The URL of the Sharepoint server.

        Raises:
            AssertionError: If the connection to the Sharepoint server fails.

        """
        context = ssl.create_default_context()
        der_certs = context.get_ca_certs(binary_form=True)
        pem_certs = [ssl.DER_cert_to_PEM_cert(der) for der in der_certs]
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as outfile:
            self.path2pem = outfile.name
            for pem in pem_certs:
                outfile.write(f"{pem}\n")

        kerberos_auth = HTTPKerberosAuth(mutual_authentication=OPTIONAL)
        self.requests = requests.get(url, auth=kerberos_auth, verify=self.path2pem)
        
        # print(self.requests.status_code)
        # print(self.requests.headers)
        assert self.requests.ok, 'Cannot connect the sharepoint server. Please check your credentials.'
        
    @property
    def content(self):
        return self.requests.content
    
    def save(self, name):
        
        with open(name, "wb") as f:
            f.write(self.content)
            
        return self
    
    def close(self):
        
        os.unlink(self.path2pem) 


if __name__ == '__main__':

    import pandas as pd
    from io import BytesIO
    url = "https://sites.inside-share3.bosch.com/sites/144527/Documents/03_OnSpecification/24-QAT_CustomerRequiremt/Daimler_MGU/Load_Spectrum/V2_MBC-SK/MAT-Files/Fahrer1/Alb650/Alb650.mat"

    r = SharepointAccess(url)


    with open("test.mat", "wb") as f:
        f.write(r.content)




